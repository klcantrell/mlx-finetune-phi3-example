import re
import json
from typing import List
from dataclasses import dataclass

import instructor
import pytest
from playwright.sync_api import Page
from bs4 import BeautifulSoup
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = instructor.from_openai(OpenAI())


@dataclass
class ExtractedHtml:
    with_ingredients: str
    before_ingredients: str


urls = [
    "https://www.favfamilyrecipes.com/musubi/",
    "https://www.foodiecrush.com/my-moms-homemade-spaghetti-and-meat-sauce-recipe/",
    "https://playswellwithbutter.com/spam-musubi-recipe/",
    "https://www.allrecipes.com/recipe/158140/spaghetti-sauce-with-ground-beef/",
    "https://soulandstreusel.com/hawaiian-fried-garlic-chicken/",
    "https://www.thecookierookie.com/best-twice-baked-potatoes-recipe/",
    "https://southernbite.com/chicken-and-sausage-gumbo/",
    "https://hungryinthailand.com/thai-garlic-pepper-chicken/",
    "https://thisishowicook.com/thai-garlic-pepper-chicken-recipe/",
    "https://hot-thai-kitchen.com/garlic-pepper-chicken/",
    "https://pupswithchopsticks.com/pad-woon-sen/",
    "https://www.cookingwithnart.com/easy-thai-style-omelet-kai-jeow/",
    "https://www.seriouseats.com/food-lab-creamy-cheesy-ultimate-spinach-lasagna-recipe",
    "https://natashaskitchen.com/lasagna-recipe/",
    "https://www.thechunkychef.com/copycat-skyline-cincinnati-chili/",
    "https://www.browneyedbaker.com/all-american-beef-chili/",
    "https://broccyourbody.com/tom-kha-inspired-soup-with-crispy-tofu/",
    "https://sugarspunrun.com/egg-salad-recipe",
    "https://www.budgetbytes.com/louisiana-red-beans-rice/",
    "https://mykitchenserenity.com/quick-easy-red-beans-rice-sausage/",
]

system_prompt = "You will be provided HTML from a recipe website. Extract the ingredients from the HTML. Include the quantity with each ingredient name if it is present. If there are no ingredients, respond with 'none'.\n\nRecipe input:"


@pytest.mark.parametrize("url", urls)
def test_scrape(page: Page, url: str):
    print(f"[START] {url}")

    page.goto(url)
    page_source = page.content()
    text_input = _extract_rough(page_source)

    response_with_ingredients = client.chat.completions.create(
        model="gpt-4o",
        response_model=Recipe,
        messages=[
            {
                "role": "user",
                "content": f"{system_prompt}\n\n{text_input.with_ingredients}",
            },
        ],
    )
    response_with_ingredients.ingredients = list(
        map(lambda r: r.replace(",", ""), response_with_ingredients.ingredients)
    )

    with open("data/train_positive.jsonl", "a") as train_positive_file, open(
        "data/train_negative.jsonl", "a"
    ) as train_negative_file:
        if len(response_with_ingredients.ingredients) == 0:
            train_negative_file.write(
                json.dumps(
                    {
                        "text": f"<|user|>{system_prompt}\n{text_input.with_ingredients}<|end|>\n<|assistant|>\nnone<|end|>"
                    }
                )
                + "\n"
            )
        else:
            train_positive_file.write(
                json.dumps(
                    {
                        "text": f"<|user|>{system_prompt}\n{text_input.with_ingredients}<|end|>\n<|assistant|>\n{', '.join(response_with_ingredients.ingredients)}<|end|>"
                    }
                )
                + "\n"
            )

        train_negative_file.write(
            json.dumps(
                {
                    "text": f"<|user|>{system_prompt}\n{text_input.before_ingredients}<|end|>\n<|assistant|>\nnone<|end|>"
                }
            )
            + "\n"
        )

    print(f"[END] {url}")


def _extract_rough(source: str, width: int = 2500) -> ExtractedHtml:
    soup = BeautifulSoup(source, "html.parser")
    for element in soup.select(
        "svg, img, style, script, iframe, header, nav, form, footer"
    ):
        element.decompose()

    for tag in soup.find_all(True):
        if len(tag.get_text(strip=True)) == 0:
            tag.decompose()
        tag.attrs = {}

    html = str(soup.find("body"))
    ingredients_headings = list(
        re.finditer(r"<h(?:2|3|4)>\s*ingredients", html, re.IGNORECASE)
    )

    if len(ingredients_headings) < 1:
        raise Exception("This site does not seem to have ingredients")

    first_ingredients_heading = ingredients_headings[0]
    first_ingredients_heading_index = first_ingredients_heading.span()[0]

    last_ingredients_heading = ingredients_headings[-1]
    last_ingredients_heading_index = last_ingredients_heading.span()[0]

    return ExtractedHtml(
        before_ingredients=html[: min(first_ingredients_heading_index, width)],
        with_ingredients=html[
            last_ingredients_heading_index : last_ingredients_heading_index + width
        ],
    )


class Recipe(BaseModel):
    ingredients: List[str]
