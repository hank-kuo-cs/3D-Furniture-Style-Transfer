from .furniture import Furniture


class TripletFurniture:
    def __init__(self,
                 sample_furniture: Furniture,
                 positive_furniture: Furniture,
                 negative_furniture: Furniture):

        self._triplet_furniture = [sample_furniture, positive_furniture, negative_furniture]

        self.check_members()

    def __getitem__(self, item):
        return self._triplet_furniture[item]

    @property
    def sample_furniture(self):
        return self._triplet_furniture[0]

    @property
    def positive_furniture(self):
        return self._triplet_furniture[1]

    @property
    def negative_furniture(self):
        return self._triplet_furniture[2]

    def check_members(self):
        assert isinstance(self.sample_furniture, Furniture)
        assert isinstance(self.positive_furniture, Furniture)
        assert isinstance(self.negative_furniture, Furniture)
