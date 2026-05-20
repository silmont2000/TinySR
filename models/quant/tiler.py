import torch


def gaussian_weights(tile_width, tile_height, nbatches, in_channels, device, dtype):
    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    midpoint_y = tile_height / 2

    x = torch.arange(tile_width, device=device, dtype=torch.float32)
    y = torch.arange(tile_height, device=device, dtype=torch.float32)

    x_probs = torch.exp(-((x - midpoint_x) ** 2) /
                        (tile_width * tile_width) / (2 * var))
    y_probs = torch.exp(-((y - midpoint_y) ** 2) /
                        (tile_height * tile_height) / (2 * var))

    weights = torch.outer(y_probs, x_probs)
    weights = weights.to(dtype=dtype)
    return weights.expand(nbatches, in_channels, tile_height, tile_width)


@torch.no_grad()
def tile_sample(
    lq_latent,
    transformer,
    timesteps,
    pooled_prompt_embeds,
    weight_dtype,
    latent_tiled_size=64,
    latent_tiled_overlap=8,
):
    _, _, height, width = lq_latent.size()
    tile_size = latent_tiled_size
    tile_overlap = latent_tiled_overlap

    if height * width <= tile_size * tile_size:
        model_pred = transformer(
            hidden_states=lq_latent,
            timestep=timesteps,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        return model_pred.to(lq_latent.device, dtype=weight_dtype)

    tile_size = min(tile_size, min(height, width))
    tile_weights = gaussian_weights(
        tile_size, tile_size, 1,
        transformer.config.in_channels,
        lq_latent.device, weight_dtype,
    )

    grid_rows = 0
    cur_x = 0
    while cur_x < width:
        cur_x = max(grid_rows * tile_size -
                    tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1

    grid_cols = 0
    cur_y = 0
    while cur_y < height:
        cur_y = max(grid_cols * tile_size -
                    tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1

    noise_preds = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if row == grid_rows - 1:
                ofs_x = width - tile_size
            else:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
            if col == grid_cols - 1:
                ofs_y = height - tile_size
            else:
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
            input_tile = lq_latent[:, :, ofs_y: ofs_y +
                                   tile_size, ofs_x: ofs_x + tile_size]
            pred = transformer(
                hidden_states=input_tile.to(
                    lq_latent.device, dtype=weight_dtype),
                timestep=timesteps,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            noise_preds.append(pred)

    noise_pred = torch.zeros(
        lq_latent.shape, device=lq_latent.device, dtype=weight_dtype)
    contributors = torch.zeros(
        lq_latent.shape, device=lq_latent.device, dtype=weight_dtype)

    for row in range(grid_rows):
        for col in range(grid_cols):
            if row == grid_rows - 1:
                ofs_x = width - tile_size
            else:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
            if col == grid_cols - 1:
                ofs_y = height - tile_size
            else:
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
            index = row * grid_cols + col
            noise_pred[:, :, ofs_y: ofs_y + tile_size, ofs_x: ofs_x +
                       tile_size] += noise_preds[index] * tile_weights
            contributors[:, :, ofs_y: ofs_y + tile_size,
                         ofs_x: ofs_x + tile_size] += tile_weights

    model_pred = noise_pred / contributors.clamp_min(1e-8)
    return model_pred.to(lq_latent.device, dtype=weight_dtype)
