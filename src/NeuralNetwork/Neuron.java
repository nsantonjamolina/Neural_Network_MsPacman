package NeuralNetwork;

import java.util.Random;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class Neuron {

    //El ultimo peso es el del bias
    private float[] _weights;
    private float _exit;

    private static int bias = 1;

    public Neuron(int inputs) {

        _weights = new float[inputs + 1];

        for (float weight : _weights) {
            weight = randomFloat((-1), 1);
        }
    }

    public float[] multiplication(float[] _inputs) {

        float[] products = new float[_inputs.length + 1];

        for(int i = 0; i < _inputs.length; i++) {
            products[i]  = _inputs[i] * _weights[i];
        }

        products[_inputs.length] = bias * _weights[_inputs.length];

        return products;
    }

    public float sumatory(float[] inputs) {

        float sumatory = 0;
        float[] products = multiplication(inputs);

        for(float product : products) {
            sumatory += product;
        }

        return sumatory;
    }

    public float sigmoidalActivation(float sumatory) {
        return  (float) (1/(1 + Math.pow(Math.E, -sumatory)));
    }

    public float feedForward(float[] inputs) {
        _exit = sigmoidalActivation(sumatory(inputs));
        return _exit;
    }

    public float getExit() {
        return _exit;
    }

    public static float randomFloat(float min, float max) {
        Random random = new Random();
        return (random.nextFloat() * (min - max) + min);
    }
}
