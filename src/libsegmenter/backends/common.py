def compute_num_segments(num_samples, hop_size, segment_size):
    return (num_samples // hop_size) - (segment_size // hop_size) + 1


def compute_num_samples(num_segments, hop_size, segment_size):
    return (num_segments - 1) * hop_size + segment_size
