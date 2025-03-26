from pydantic import BaseModel, Field, conint

class RecipeRating(BaseModel):
    rating: int = Field(description="Rating (1-5)")
