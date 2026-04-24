"""Check CSV structure and reviewer_comments content."""
import csv
import json

with open('data/deepreview/test_2024.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    row = next(reader)

print("CSV columns:", list(row.keys()))
print()
print(" reviewer_comments field exists:", "reviewer_comments" in row)
if "reviewer_comments" in row:
    rc = row["reviewer_comments"]
    print(" reviewer_comments length:", len(rc))
    print(" reviewer_comments first 500 chars:")
    print(rc[:500])
    print()
    try:
        parsed = json.loads(rc)
        print(" Successfully parsed as JSON")
        if isinstance(parsed, list) and parsed:
            print(" Number of reviews:", len(parsed))
            print(" First review keys:", list(parsed[0].keys()))
            if "content" in parsed[0]:
                print(" content keys:", list(parsed[0]["content"].keys()))
    except Exception as e:
        print(" Failed to parse as JSON:", e)
