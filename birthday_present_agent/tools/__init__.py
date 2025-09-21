"""Custom tools for the birthday present agent."""

from .shopping import shopping_search
from .product_details import fetch_product_details
from .social_profile import fetch_social_profile

__all__ = [
    "shopping_search",
    "fetch_product_details",
    "fetch_social_profile",
]
