class WhoWroteMe:
    models = [];
    modelStatistics = {};
    modelsDir = "";

    # Load a directory with learning resourses, run a lexicographical analysis on it and store the results
    # Directory tree format author/text.txt
    def loadAndAnalyzeDir(self, directory, modelStatistics): pass

    # Load a file, analyze it and return the statistics for it
    def loadAndAnalyzeFile(self, file): pass

    # Make prediction for text author with current model - returns author
    def makePrediction(self, model, statistics): pass

    # Load premade models from a directory
    def loadModels(self): pass

    # Select model based on hints
    def selectModel(self): pass

    # Generate a model from with given statistics + metadata
    def generateModel(self, learningData, modelStatistics): pass

class Model:
    # Creates a model based on a given learning data and statistics
    def createModel(self, learningData, modelStatistics): pass

    # Export model as a file
    def exportModel(self, file): pass

    # Import model from a file
    def importModel(self, file): pass

    # Make prediction with a given model and statistics
    def makePrediction(self, textStatistics): pass

    # <Obj> model // type depends on library

# I haven't decided which machine learning algorithm to use yet. This is still open for research.
