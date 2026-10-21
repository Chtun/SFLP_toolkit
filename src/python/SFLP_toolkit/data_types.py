"""Data types module for spherical tiling and heatmap visualizations.

This module defines the core data types used in spherical tiling and heatmap visualizations.
It includes classes for representing points in different coordinate systems (pixel, spherical,
and Cartesian) and utilities for validation.

Classes:
    SpatialError: Base exception class for spatial-related errors.
    ValidationError: Exception for validation errors in spatial data.
    Point: Represents a point in pixel coordinates.
    GeographicPoint: Represents a point in geographic coordinates (longitude/latitude).
    Vector: Represents a point in 3D Cartesian coordinates.
"""

from dataclasses import dataclass
from typing import Union, Tuple, List
import numpy as np
import re



class ValidationError(BaseException):
    """Exception raised when data/argument validation fails."""
    pass


@dataclass(frozen=True)
class Point:
    """Represents a point in pixel coordinates.
    
    This class represents a point in a 2D pixel coordinate system, typically
    used for viewport center trajectories in video frames.
    
    Attributes:
        pixel_x (int): X coordinate in pixels.
        pixel_y (int): Y coordinate in pixels.
    
    Raises:
        ValidationError: If pixel coordinates are negative.
    """
    x: float
    y: float

    def __post_init__(self) -> None:
        """Validates point coordinates after initialization."""
    
    def as_tuple(self) -> Tuple[float, float]:
        """Returns the point coordinates as a tuple.
        
        Returns:
            Tuple[float, float]: A tuple of (x, y) coordinates.
        """
        return (self.x, self.y)

@dataclass(frozen=True)
class GeographicPoint:
    """Represents a point in geographic coordinates.
    
    This class represents a point on a sphere using longitude and latitude coordinates,
    typically used for representing viewing directions in 360-degree videos.
    
    At lon = 0 and lat = 0, the equivalent vector would be (x=1, y=0, z=0).

    Attributes:
        lon (float): Longitude in degrees, range [-180, 180].
        lat (float): Latitude in degrees, range [-90, 90].
    
    Raises:
        ValidationError: If coordinates are outside their valid ranges.
    """
    lon: float
    lat: float

    def __post_init__(self) -> None:
        """Validates geographic coordinates after initialization."""
        if not -180 <= self.lon <= 180:
            raise ValidationError("Longitude must be between -180 and 180 degrees")
        if not -90 <= self.lat <= 90:
            raise ValidationError("Latitude must be between -90 and 90 degrees")

    def normalize_coordinates(self) -> 'GeographicPoint':
        """Normalizes coordinates to standard ranges.
        
        Ensures longitude is in [-180, 180] and latitude is in [-90, 90].
        
        Returns:
            GeographicPoint: A new GeographicPoint with normalized coordinates.
        """
        normalized_lon = ((self.lon + 180) % 360) - 180
        normalized_lat = ((self.lat + 90) % 180) - 90
        return GeographicPoint(normalized_lon, normalized_lat)
    
    def as_tuple(self) -> Tuple[float, float]:
        """Returns the point coordinates as a tuple.
        
        Returns:
            Tuple[float, float]: A tuple of (longitude, latitude) coordinates.
        """
        return (self.lon, self.lat)


@dataclass(frozen=True)
class Vector:
    """Represents a point in 3D Cartesian coordinates.
    
    This class represents a point in 3D space using Cartesian coordinates,
    typically used as a unit vector representing a viewing direction.
    
    Attributes:
        x (float): X coordinate. The X axis is the front-back axis.
        y (float): Y coordinate. The Y axis is the left-right axis.
        z (float): Z coordinate. The Z axis is the up-down axis.
    
    Raises:
        ValidationError: If the vector has zero length.
    """
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        """Validates vector after initialization."""
        if self.length() == 0:
            raise ValidationError("Vector cannot have zero length")

    def length(self) -> float:
        """Calculates the length (magnitude) of the vector.
        
        Returns:
            float: The length of the vector.
        """
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> 'Vector':
        """Returns a normalized (unit length) version of the vector.
        
        Returns:
            Vector: A new Vector with unit length.
        
        Raises:
            ValidationError: If the vector has zero length.
        """
        length = self.length()
        if length == 0:
            raise ValidationError("Cannot normalize zero-length vector")
        return Vector(
            x=self.x / length,
            y=self.y / length,
            z=self.z / length
        )
    
    def dot(self, other_vector) -> 'float':
        """
        Calculates the dot product of this vector with another vector.

        Args:
            other_vector (Vector): The other vector to compute the dot product with.

        Returns:
            float: The scalar result of the dot product.
        """
        if not isinstance(other_vector, Vector):
            raise TypeError("The other object must be a Vector instance.")

        return self.x * other_vector.x + self.y * other_vector.y + self.z * other_vector.z
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Returns the vector coordinates as a tuple.
        
        Returns:
            Tuple[float, float, float]: A tuple of (x, y, z) coordinates.
        """
        return (self.x, self.y, self.z)
    
    def round(self, decimals: int, round_down: bool = False):
        # Return a new instance with rounded values
        if round_down:
            factor = 10**decimals
            rounded_x = np.floor(self.x * factor) / factor
            rounded_y = np.floor(self.y * factor) / factor
            rounded_z = np.floor(self.z * factor) / factor
        else:
            rounded_x = np.round(self.x, decimals=decimals)
            rounded_y = np.round(self.y, decimals=decimals)
            rounded_z = np.round(self.z, decimals=decimals)

        return Vector(
            x=rounded_x,
            y=rounded_y,
            z=rounded_z
        )

    def to_spherical(self) -> GeographicPoint:
        _lon = float(round(np.degrees(np.arctan2(self.y, self.x)), 6))
        _lat = float(round(np.degrees(np.arcsin(self.z / np.sqrt(self.x**2 + self.y**2 + self.z**2))), 6))

        # Ensure longitude is in [-180, 180]
        if _lon > 180:
            _lon -= 360
        elif _lon <= -180:
            _lon += 360

        return GeographicPoint(
                lon=_lon,
                lat=_lat
            )
    
    @classmethod
    def from_string(cls, vector_str: str):
        match = re.match(r"Vector\(x=([\d\.\-e]+), y=([\d\.\-e]+), z=([\d\.\-e]+)\)", vector_str)
        if match:
            x, y, z = match.groups()
            return Vector(float(x), float(y), float(z))
        else:
            raise ValueError(f"Invalid vector string format: {vector_str}")

    @classmethod
    def from_geographic(cls, lon: float, lat: float) -> 'Vector':
        """Creates a unit vector from geographic coordinates.
        
        Args:
            lon (float): Longitude in degrees.
            lat (float): Latitude in degrees.
        
        Returns:
            Vector: A new unit Vector representing the direction.
        
        Raises:
            ValidationError: If the coordinates are invalid.
        """
        # Validate inputs first
        if not -180 <= lon <= 180:
            raise ValidationError("Longitude must be between -180 and 180 degrees")
        if not -90 <= lat <= 90:
            raise ValidationError("Latitude must be between -90 and 90 degrees")

        # Convert to radians
        theta = np.radians(lon)
        phi = np.radians(90 - lat)  # Convert to co-latitude

        # Calculate cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        return cls(
            x=float(x),
            y=float(y),
            z=float(z)
        )


def convert_vectors_to_coordinates(
    vectors: Union[List[Vector], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a collection of vectors to plottable coordinates.
    
    This function takes a list or array of Vector objects and converts them 
    to longitude and latitude coordinates suitable for plotting. It handles
    both single vectors and collections of vectors.
    
    Args:
        vectors: List or array of Vector objects to convert.
            
    Returns:
        Tuple containing:
        - np.ndarray: Array of longitude coordinates
        - np.ndarray: Array of latitude coordinates
            
    Raises:
        ValidationError: If input is empty or contains invalid vectors.
        
    Example:
        >>> vectors = [Vector(1, 0, 0), Vector(0, 1, 0)]
        >>> lons, lats = convert_vectors_to_coordinates(vectors)
        >>> print(lons, lats)
        [0.0, 90.0] [0.0, 0.0]
    """
    if not vectors:
        raise ValidationError("Empty vector collection provided")
        
    try:
        # Convert to list if numpy array
        vector_list = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors
        
        # Validate input types
        if not all(isinstance(v, Vector) for v in vector_list):
            raise ValidationError("All elements must be Vector instances")
        
        # Convert each vector to geographic_point coordinates
        geographic_point_points = [
            GeographicPoint(
                lon=np.degrees(np.arctan2(v.y, v.x)),
                lat=np.degrees(np.arcsin(v.z / np.sqrt(v.x**2 + v.y**2 + v.z**2)))
            )
            for v in vector_list
        ]
        
        # Extract coordinates
        lons = np.array([p.lon for p in geographic_point_points])
        lats = np.array([p.lat for p in geographic_point_points])
        
        # Ensure longitude is in [-180, 180]
        lons = np.where(lons > 180, lons - 360, lons)
        lons = np.where(lons <= -180, lons + 360, lons)
        
        return lons, lats
        
    except Exception as e:
        raise ValidationError(f"Failed to convert vectors to coordinates: {str(e)}")
