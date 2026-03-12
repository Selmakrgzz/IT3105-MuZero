import pandas as pd
import requests

API_KEY = "oabkAbIDTBvHuURBfR8EaJBvenPlFDQiVMwvqyz8"

df = pd.read_csv("movies.csv")

results = []

for title in df["title"]:
    search_url = f"https://api.watchmode.com/v1/search/?apiKey={API_KEY}&search_field=name&search_value={title}"

    r = requests.get(search_url).json()

    disney = False

    if r["title_results"]:
        movie_id = r["title_results"][0]["id"]

        sources_url = f"https://api.watchmode.com/v1/title/{movie_id}/sources/?apiKey={API_KEY}"
        sources = requests.get(sources_url).json()

        for s in sources:
            if "disney" in s["name"].lower():
                disney = True
                break

    results.append({"title": title, "on_disney_plus": disney})

out = pd.DataFrame(results)
out.to_csv("disney_check.csv", index=False)