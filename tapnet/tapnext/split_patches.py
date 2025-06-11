import torch

class PatchSplitter:
    def __init__(self, def_shape, original_image_dims):
        self.def_shape = def_shape
        self.original_image_dims = original_image_dims
        self.tracks_mask = None
        self.max_patch_queries = None

    def split_video(self, frame):
        _, H, W, _ = frame.shape
        patches = []
        h, w = self.def_shape
        a = ((W/2)-(w/2))
        b = ((W/2)+(w/2))
        patches.append(frame[:,:h, :w])
        patches.append(frame[:,:h, a:b])
        patches.append(frame[:,:h, W-w:])
        patches.append(frame[:,H-h:, :w])
        patches.append(frame[:,H-h:, a:b])
        patches.append(frame[:,H-h:, W-w:])

        return torch.cat(patches, dim=0)

    def split_removed_indices(self, removed_indices):
        if self.max_patch_queries is None:
            return torch.ones(1, device='cuda', dtype=torch.bool)
        orig_mask = torch.ones(self.tracks_mask.shape[1], device=self.tracks_mask.device, dtype=torch.bool)
        orig_mask[removed_indices] = False

        split_mask = torch.zeros(6, self.max_patch_queries, device=self.tracks_mask.device, dtype=torch.bool)  # Removing the extra padded queries every step
        for i in range(6):
            split_mask[i, :(self.tracks_mask[i]).sum()] = orig_mask[self.tracks_mask[i]]

        return split_mask

    def split_queries(self, queries, t):
        H, W = self.original_image_dims
        h, w = self.def_shape
        a = ((W/2)-(w/2))
        b = ((W/2)+(w/2))
        pad_query = torch.tensor([t, 0.0, 0.0], device=queries.device)
        mask0 = (queries[0,:,1] >= 0) & (queries[0,:,1] < w) & (queries[0,:,2] >= 0) & (queries[0,:,2] < h)
        mask1 = (queries[0,:,1] >= a) & (queries[0,:,1] < b) & (queries[0,:,2] >= 0) & (queries[0,:,2] < h)
        mask2 = (queries[0,:,1] >= W-w) & (queries[0,:,1] < W) & (queries[0,:,2] >= 0) & (queries[0,:,2] < h)
        mask3 = (queries[0,:,1] >= 0) & (queries[0,:,1] < w) & (queries[0,:,2] >= H-h) & (queries[0,:,2] < H)
        mask4 = (queries[0,:,1] >= a) & (queries[0,:,1] < b) & (queries[0,:,2] >= H-h) & (queries[0,:,2] < H)
        mask5 = (queries[0,:,1] >= W-w) & (queries[0,:,1] < W) & (queries[0,:,2] >= H-h) & (queries[0,:,2] < H)
        self.tracks_mask = torch.stack([mask0, mask1, mask2, mask3, mask4, mask5], dim=0)  # 6,N (N is num queries in this timestep - changes every step)

        self.max_patch_queries = max(mask0.sum(), mask1.sum(), mask2.sum(), mask3.sum(), mask4.sum(), mask5.sum()).item()
        transforms = torch.tensor([[0, 0], [a, 0], [W-w, 0], [0, H-h], [a, H-h], [W-w, H-h]], device=queries.device)
        patch_queries = pad_query.repeat(6, self.max_patch_queries, 1)
        for i in range(6):
            patch_queries[0, :self.tracks_mask[i].sum(), :] = queries[0, self.tracks_mask[i], :] - transforms[i]

        return patch_queries

    def combine_tracks(self, tracks, status):
        N = self.tracks_mask.shape[1]
        H, W = self.original_image_dims
        h, w = self.def_shape
        a = ((W/2)-(w/2))
        b = ((W/2)+(w/2))
        transforms = torch.tensor([[0, 0], [a, 0], [W-w, 0], [0, H-h], [a, H-h], [W-w, H-h]], device=tracks.device)

        combined_tracks = torch.zeros(6, N, 2, device=tracks.device)  # 6,N,2
        combined_status = torch.zeros(6, N, device=status.device)  # 6,N
        for i in range(6):
            combined_tracks[i, self.tracks_mask[i], :] = tracks[i, :self.tracks_mask[i].sum(), :] + transforms[i]
            combined_status[i, self.tracks_mask[i]] = status[i, :self.tracks_mask[i].sum()]

        row_idx = combined_status.argmax(dim=0)
        final_status = combined_status.max(dim=0, keepdim=True).values
        tracks_perm = combined_tracks.permute(1, 0, 2)  # (N, B, 2)
        index = row_idx[:, None].expand(N, 1).unsqueeze(-1).expand(N, 1, 2)
        final_tracks = tracks_perm.gather(1, index)  # (N, 1, 2)
        final_tracks = final_tracks.squeeze(1).unsqueeze(0)

        return final_tracks, final_status