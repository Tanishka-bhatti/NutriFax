from ultralytics import YOLO
import pandas as pd

# Load trained YOLO model
model = YOLO("C:/food-project/runs/detect/food_detector2/weights/best.pt")

# Load nutrition table
nutrition = pd.read_csv("C:/food-project/nutrition_filtered.csv")

# Run detection on a new test image
results = model.predict("C:/food-project/aloo_tikkit.jpeg", conf=0.5)

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])      # class ID
        dish = model.names[cls_id]    # class name
        print("Detected:", dish)

        # Match nutrition info
        row = nutrition[nutrition["Dish Name"].str.lower() == dish.lower()]
        if not row.empty:
            print("Nutrition info:", row.to_dict(orient="records")[0])
