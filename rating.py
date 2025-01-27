from pydantic import BaseModel, Field, conint

class RecipeRating(BaseModel):
    rating: conint(ge=1, le=5) = Field(description="Rating")
