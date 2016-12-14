package NeuralNetwork;

import pacman.game.util.IO;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class NeuralNetwork {

    private static String fileName = "dataset.txt";
    private static float[][] trainingDataset;
    private static float[][] validationDataset;

    private int _inputNeurons;
    private Neuron[] _hiddenLayer;
    private Neuron _outputLayer;

    private float[][] _weightIncrementsIH;
    private float[] _weightIncrementsHO;

    private float _learningRate = 0.2f;

    private int _epochs;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, int percentage) {

        _inputNeurons = inputNeurons;
        _weightIncrementsIH = new float[hiddenNeurons][_inputNeurons + 1];
        _weightIncrementsHO = new float[hiddenNeurons + 1];
        _epochs = 0;

        _hiddenLayer = new Neuron[hiddenNeurons];

        for(int i = 0; i < hiddenNeurons; i++) {
            _hiddenLayer[i] = new Neuron(_inputNeurons, LayerType.HIDDEN);
        }

        _outputLayer = new Neuron(hiddenNeurons, LayerType.OUTPUT);

        if(trainingDataset == null) {
            LoadPacmanData(percentage);
        }
    }

    public void Train() {
        BackPropagation(trainingDataset, validationDataset, _learningRate);
    }

    private void BackPropagation(float[][] trainingDataset, float[][] validationDataset, float learningRate) {

        float error = Float.MAX_VALUE;
        float lastEcm = Float.MAX_VALUE;
        float ecm = Float.MAX_VALUE;

        // Repetir mientras no se cumpla condición de parada
        do {
            lastEcm = ecm;
            error = 0;

            float[] inputs = new float[trainingDataset[0].length - 1];
            float expetedOutput;

            for(float[] tuple : trainingDataset) {
                //Inicializar Δwih y Δwho a 0
                _weightIncrementsIH = new float[_hiddenLayer.length][_hiddenLayer[0].getWeights().length];
                _weightIncrementsHO = new float[_outputLayer.getWeights().length];

                // Entradas de la tupla y salida esperad
                int tupleLength = trainingDataset[0].length - 1;
                for(int i = 0; i < tupleLength; i++) {
                    inputs[i] = tuple[i];
                }
                expetedOutput = tuple[tupleLength];

                //calcular la salida de la red neuronal (feedfordward)
                float output = forwardPropagation(inputs);

                //calcular los errores de la capa de salida
                _outputLayer.calculateError(expetedOutput);

                //calcular Δwho para los pesos de todas conexiones entre
                //la capa oculta y la capa de salida (who)
                for (int i = 0; i < _hiddenLayer.length; i++) {
                    float outputError = _outputLayer.getError();
                    float hiddenNeuronExit = _hiddenLayer[i].getExit();
                    _weightIncrementsHO[i] += _learningRate * outputError * hiddenNeuronExit;
                }
                _weightIncrementsHO[_hiddenLayer.length] += _learningRate * _outputLayer.getError() * Neuron.getBias();

                //calcular los errores de la capa oculta
                for (int i = 0; i < _hiddenLayer.length; i++) {
                    float outputError = _outputLayer.getError();
                    float weightHO = _outputLayer.getWeights()[i];
                    float propagatedError = outputError * weightHO;
                    _hiddenLayer[i].calculateError(propagatedError);
                }

                //calcular Δwih para los pesos de todas conexiones entre
                //la capa de entrada y la capa oculta(wih).
                for (int i = 0; i < _hiddenLayer.length; i++) {
                    for(int j = 0; j < inputs.length; j++) {
                        _weightIncrementsIH[i][j] += _learningRate * _hiddenLayer[i].getError() * inputs[j];
                    }
                    _weightIncrementsIH[i][inputs.length] += _learningRate * _hiddenLayer[i].getError() * Neuron.getBias();
                }

                // Actualizar los pesos de la red who y wih
                UpdateWeights();
            }

            // Pasamos la red neuronal por el dataset de validación para calcular el error cuadrático
            for(float[] tuple : validationDataset) {
                int tupleLength = validationDataset[0].length - 1;
                for(int i = 0; i < tupleLength; i++) {
                    inputs[i] = tuple[i];
                }
                expetedOutput = tuple[tupleLength];

                float output = forwardPropagation(inputs);
                float diference = expetedOutput - output;

                error += Math.pow(diference,2);
            }

            _epochs++; // Contamos las epochs del entrenamiento
            ecm = error/validationDataset.length; // Calculamos el error cuadrático medio dividiendo el error cuadrático entre el length del dataset de validación
            System.out.println("Epoch amount: " + _epochs + " - ECM: " + ecm + " - Last ecm " + lastEcm);
        } while (ecm < lastEcm);
        System.out.println("--------END TRAINING--------");
    }

    public float forwardPropagation (float [] _inputLayer) {

        float[] hiddenLayerExits = new float[_hiddenLayer.length];

        for(int i = 0; i < _hiddenLayer.length; i++) {
            hiddenLayerExits[i] = _hiddenLayer[i].feedForward(_inputLayer);
        }

        return _outputLayer.feedForward(hiddenLayerExits);
    }

    private void UpdateWeights() {
        // Hacemos update del peso del bias de cada neurona de la capa oculta y de la neurona de salida
        for (int i = 0; i < _outputLayer.getWeights().length; i++) {
            if(_weightIncrementsHO[i] != 0){
                float weight = _outputLayer.getWeights()[i];
                float weightIncrement = _weightIncrementsHO[i];
                float value = weight + weightIncrement;
                _outputLayer.setWeight(i,value);
            }
        }

        for (int i = 0; i < _hiddenLayer.length; i++) {
            for(int j = 0; j < _hiddenLayer[i].getWeights().length; j++) {
                if(_weightIncrementsIH[i][j] != 0) {
                    float weight = _hiddenLayer[i].getWeights()[j];
                    float weightIncrement = _weightIncrementsIH[i][j];
                    float value = weight + weightIncrement;
                    _hiddenLayer[i].setWeight(j,value);
                }
            }
        }
    }

    private void LoadPacmanData (int trainingDatasetPercentage) {
        String data = IO.loadFile(fileName);
        String[] dataLines = data.split("\n");

        String[] lineSplited = dataLines[0].split(",");
        int datasetLength = dataLines.length * trainingDatasetPercentage/100;

        trainingDataset = new float[datasetLength][lineSplited.length];
        validationDataset = new float[dataLines.length - datasetLength][lineSplited.length];

        for(int i = 0; i < dataLines.length; i++) {

            lineSplited = dataLines[i].split(",");

            for (int j = 0; j < lineSplited.length; j++) {
                if(i < trainingDataset.length) {
                    trainingDataset[i][j] = Float.parseFloat((lineSplited[j]));
                } else {
                    validationDataset[i%trainingDataset.length][j] = Float.parseFloat((lineSplited[j]));
                }
            }
        }
    }
}
