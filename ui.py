import model
import stylometry
import os 

UIS_LOCATION = './ui/'
# This will be PyQT4 one day yay.

class ui:
    def __init__(self, models, text_analysis):
        self._models = models
        self._text_analysis = text_analysis
        self.initialize_menus()
        self.initialize_menu_choices()
        self.main_menu()
    
    def main_menu(self):
        print(self._main_menu)
        choice = input()
        self._main_menu_choices[int(choice)]()

    def prediction_menu(self): 
        print(self._prediction_menu)
        choice = input()
        self._prediction_menu_choices[int(choice)]()

    def expand_library_menu(self):
        print(self._library_menu)
        choice = input()
        self._library_menu_choices[int(choice)]()

    def exit(self):
        print("Goodbye!")

    def prompt_input(): pass

    def check_bad_input(self): pass

    def add_author(self): pass

    def predict(self): pass
        #display algo choice
        #input file then

    def display_accuracy(self): pass

    def display_authors(self): pass

    def initialize_menus(self):
        self._main_menu = self.load_menu('main_menu.txt')
        self._library_menu = self.load_menu('library_menu.txt')
        self._prediction_menu = self.load_menu('prediction_menu.txt')
        self._algorythm_menu = self.load_menu('algorythm_menu.txt')

    def initialize_menu_choices(self):
        self._main_menu_choices = {1: self.prediction_menu, 2: self.expand_library_menu, 3: self.display_accuracy, 4: self.exit}
        self._library_menu_choices = {1: self.add_author, 2: self.display_authors, 3: self.main_menu, 4: self.exit}
        self._prediction_menu_choices = {1: self.predict, 2: self.main_menu, 3: self.exit}

    def load_menu(self, menu_name):
        #TODO Error handling
        menu_location = os.path.join(UIS_LOCATION, menu_name)
        menu_file = open(menu_location, 'r')
        menu = menu_file.read()
        menu_file.close()
        return menu
        


print("Welcome to Who Wrote Me v0.1.0")
text_analysis = stylometry.Text_Analysis()
models = model.Models(text_analysis.text_features_library)

u = ui(models, text_analysis)






