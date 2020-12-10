class Coordinate:
    lat: float
    lng: float

    def __init__(self, lng: float, lat: float):
        self.lat = lat
        self.lng = lng

    def shape(self):
        return [self.lng, self.lat]

    def __repr__(self):
        return f"Longitude: {self.lng}, Latitude: {self.lat}"
