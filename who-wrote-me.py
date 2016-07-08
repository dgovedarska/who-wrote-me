import ui
import stylometry
import model
import accuracy

print("Welcome to Who Wrote Me v0.1.0")
text_analysis = stylometry.Text_Analysis()
models = model.Models(text_analysis.text_features_library)
accuracy = accuracy.accuracy()

u = ui.Ui(models, text_analysis, accuracy)