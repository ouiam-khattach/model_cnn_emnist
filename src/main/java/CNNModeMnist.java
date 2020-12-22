import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CNNModeMnist {
    public static void main(String[] args) throws IOException, InterruptedException {
        long seed=123;//exemple de random(123)prendre meme serie dans chaque execution
        double learningRate=0.001;
        long height=28;
        long width=28;
        long depth=1;
        int outputSize=10;
        int batchSize=54;
        MultiLayerConfiguration multiLayerConfiguration=new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .list()
                .setInputType(InputType.convolutionalFlat(height,width,depth))

                .layer(0,new ConvolutionLayer.Builder()
                        .nIn(depth)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .kernelSize(5,5)//filtre
                        .stride(1,1)//glisser verticalement,horiz
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2,new ConvolutionLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .kernelSize(5,5)
                        .stride(1,1)
                        .build())
                .layer(3,new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4,new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(5,new OutputLayer.Builder()
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();
        //System.out.println(multiLayerConfiguration.toJson());
        /*creation du model*/

        MultiLayerNetwork model =new MultiLayerNetwork(multiLayerConfiguration);
        model.init();
        System.out.println("Model training...");
        String path= System.getProperty("user.home")+"/mnist_png";
        System.out.println(path);
        File fileTrain=new File(path +"/training");
        FileSplit fileSplit = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS,new Random(seed));
        RecordReader recordReaderTrain=new ImageRecordReader(height,width,depth,new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplit);

        DataSetIterator dataSetIterator=new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outputSize);
        DataNormalization scaler=new ImagePreProcessingScaler(0,1);
        dataSetIterator.setPreProcessor(scaler);

        UIServer uiServer=UIServer.getInstance();
        StatsStorage statsStorage=new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        int numEpoch=1;
        for(int i=0;i<numEpoch;i++){
          model.fit(dataSetIterator);

        }
        System.out.println("Model Evaluation");
        File fileTest=new File(path +"/testing");
        FileSplit fileSplitTest = new FileSplit(fileTest, NativeImageLoader.ALLOWED_FORMATS,new Random(seed));
        RecordReader recordReaderTest=new ImageRecordReader(height,width,depth,new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSplitTest);
        DataSetIterator dataSetIteratorTest=new RecordReaderDataSetIterator(recordReaderTest,batchSize,1,outputSize);
        DataNormalization scalerTest=new ImagePreProcessingScaler(0,1);
        dataSetIteratorTest.setPreProcessor(scalerTest);

        Evaluation evalution=new Evaluation();
        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet=dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray targetLabels=dataSet.getLabels();


            INDArray predicted=model.output(features);
            evalution.eval(predicted,targetLabels);
        }

        System.out.println(evalution.stats());










        /*while (dataSetIterator.hasNext()){
            DataSet dataSet=dataSetIterator.next();
            INDArray features =dataSet.getFeatures();
            INDArray labels =dataSet.getLabels();
            System.out.println(features.shapeInfoToString());
            System.out.println(labels.shapeInfoToString());
            System.out.println("----------------------------");


        }*/


    //dawnload files image


    }
}
