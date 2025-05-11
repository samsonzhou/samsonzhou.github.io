import random
from collections import defaultdict, deque

class WCSS:
    def __init__(self, k, window_size, num_buckets):
        self.k = k  # number of counters to keep
        self.window_size = window_size
        self.num_buckets = num_buckets
        self.bucket_width = window_size // num_buckets
        self.current_bucket = 0
        self.buckets = [defaultdict(int) for _ in range(num_buckets)]
        self.total_counts = defaultdict(int)  # sum of counts across buckets
        self.items_seen = 0

    def _expire_bucket(self):
        expire_bucket_idx = (self.current_bucket + 1) % self.num_buckets
        expired_counts = self.buckets[expire_bucket_idx]
        
        # Subtract expired counts from total_counts
        for item, count in expired_counts.items():
            self.total_counts[item] -= count
            if self.total_counts[item] <= 0:
                del self.total_counts[item]
        
        # Clear expired bucket
        self.buckets[expire_bucket_idx].clear()
        
        self.current_bucket = expire_bucket_idx

    def update(self, item):
        self.items_seen += 1
        self.buckets[self.current_bucket][item] += 1
        self.total_counts[item] += 1

        # Check if we need to expire a bucket
        if self.items_seen % self.bucket_width == 0:
            self._expire_bucket()

        # Space-saving: evict least frequent if we exceed k counters
        if len(self.total_counts) > self.k:
            # Find item with smallest count
            min_item = min(self.total_counts, key=self.total_counts.get)
            min_count = self.total_counts[min_item]
            # Remove from total_counts
            del self.total_counts[min_item]
            # Also remove from all buckets
            for bucket in self.buckets:
                if min_item in bucket:
                    del bucket[min_item]

    def get_top_k(self):
        return sorted(self.total_counts.items(), key=lambda x: -x[1])

