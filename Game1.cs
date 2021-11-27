using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Datasets;
using Tensorflow.Keras.Layers;
using System;
using System.Linq;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using System.Threading.Tasks;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Gradients;
using System.Diagnostics;
using System.Collections.Generic;
using System.Threading;
using System.IO;

namespace TensorDotNetProject
{
    public class Game1 : Game
    {
        //TODO: compare with base tutorial
        //! If this reaches tries to run the models and instead hangs and OOMs then there may not be enough space on the disk
        //! 7.6gb needed, less will impact training speed

        #region variables

        GraphicsDeviceManager Graphics;
        SpriteBatch SpriteBatch;


        float LeakyReLU_alpha = 0.2f;


        int epochAmount = 1000;
        //int epochs = 2000; // Better effect, but longer time
        int batch_size = 16;

        //int save_every = 10;
        //int display_every = 5;
        //string inputImagePath = "D:/Pictures/Datasets/SimpleDatasets/Blue";//needs a extra subfolder between location and data
        string inputImagePath = "D:/Pictures/Datasets/mnist/MiniTrainingSet";//needs a extra subfolder between location and data

        string saveImgPath = "D:/Pictures/aGanTest/imgs";
        string saveModelPath = "D:/Pictures/aGanTest/models";

        int latent_dim = 100;

        int img_width = 28;
        int img_height = 28;
        int channels = 3;
        bool conditional = true;

        static string scale_interpolation = "bilinear"; //Supports: bilinear, nearest*.  Unknown: area, lanczos3, lanczos5, gaussian.  Unimplemented: bicubic, mitchellcubic
        int conditionalInputs = 1;//1 or 0 supported if conditional is true

        Shape img_shape;

        int generatorInputDims;

        //DatasetPass dataset;

        IDatasetV2 datasetV2;//new batched dataset

        SpriteFont font_Arial;
        Texture2D debug;

        Vector2 windowSize;

        //string deviceName = "";
        TimeSpan lastStopwatchTime = new TimeSpan(1);
        const int maxDisplayCount = 10;
        readonly List<string> DisplayText = new List<string>();
        string OutputText { 
            set
            {
                DisplayText.Add(value);
                if (DisplayText.Count > maxDisplayCount)
                    DisplayText.RemoveRange(0, DisplayText.Count - maxDisplayCount);
            } }

        Texture2D outputImageFake;
        Texture2D outputImage;


        Rescaling rescale;

        Tensor g_loss, d_loss, d_loss_real, d_loss_fake;

        Tensors fakeImgs;
        Tensor images;
        Tensor classValue;

        bool StartedTraining = false;
        int currentEpoch = 0;
        int lastDrawEpoch = 0;

        Stopwatch stopWatch = new Stopwatch();
        #endregion


        #region Loading

        public Game1()
        {
            Graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;
        }

        protected override void Initialize()
        {
            base.Initialize();

            windowSize = new Vector2(Graphics.GraphicsDevice.Viewport.Width, Graphics.GraphicsDevice.Viewport.Height);
        }

        protected override void LoadContent()
        {
            tf.enable_eager_execution();

            SpriteBatch = new SpriteBatch(GraphicsDevice);

            font_Arial = Content.Load<SpriteFont>("Arial");
            debug = Content.Load<Texture2D>("Debug1");

            PrepareData();
        }

        public void PrepareData()
        {
            img_shape = (img_width, img_height, channels);

            generatorInputDims = latent_dim + (conditional ? conditionalInputs : 0);

            if (img_height % 4 != 0 || img_width % 4 != 0)
            {
                throw new Exception("The width and height of the image must be a multiple of 4");
            }

            Directory.CreateDirectory(saveImgPath);
            Directory.CreateDirectory(saveModelPath);

            rescale = keras.layers.Rescaling(1f / 127.5f, -1f);
        }

        public void LoadDataset()
        {
            //dataset = keras.datasets.mnist.load_data();

            OutputText = "Finding images. Count: " + Directory.GetFiles(inputImagePath, "*", SearchOption.AllDirectories).Length;
            
            datasetV2 = keras.preprocessing.image_dataset_from_directory(inputImagePath, labels: "inferred", label_mode: "int", 
                color_mode: (channels == 3 ? "rgb" : "grayscale"), batch_size: batch_size, image_size: (img_width, img_height),
                validation_split: 0f, interpolation: scale_interpolation, subset: "training");
            //keras.preprocessing.text_dataset_from_directory
            datasetV2 = datasetV2.map(ApplyRescaling).repeat(-1).shuffle(1).prefetch(1);//try without prefetch too //.repeat(-1).shuffle(1)
            //var b = datasetV2.take(1);
            //Tensor dat = b.First().Item1;
            //Tensor dat2 = b.First().Item2;

            //NOTE: take and prefetch values are 'X * batchsize' so take(1) to get en entire batch
        }

        public Tensors ApplyRescaling(Tensors Input)
        {
            Tensor Input2 = Input[1];
            Tensor thing2 = rescale.Apply(Input[0]);//this breaks when only doing one side and does not scale, if applied to both sides the labels are made null
            return (thing2, Input2);
        }

        //.map(x => rescale.Apply(x))
        //public Tensors TransformTensors(Tensors input)//DEBUG
        //{
        //    tf.expand_dims(input, 3);
        //    //for (int i = 0; i < input.Length; i++)
        //    //{
        //    //    var a = input[i];
        //    //    input[i] = tf.expand_dims(input[i], 3);
        //    //    //input[i] = np.expand_dims(input[i].numpy(), 3).astype(np.float32);
        //    //}
        //    return input;
        //}

        #endregion


        #region methods

        private Tensorflow.Keras.Engine.Model Make_Generator_model()
        {
            Activation activation = null;

            Sequential model = keras.Sequential();

            //5 trainable layers max

            //Dense 1
            model.add(keras.layers.Dense((img_width / 4) * (img_height / 4) * 256 * channels, activation: activation, input_shape: generatorInputDims));//2 bracket sets added around (int / 4) operations
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));
            model.add(keras.layers.Reshape(((img_width / 4), (img_height / 4), 256 * channels)));//the first two seem to be the input divided by 4 (?)

            //Conv2D 1
            model.add(keras.layers.UpSampling2D());
            model.add(keras.layers.Conv2D(128, 3, 1, padding: "same", activation: activation));
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));

            //Conv2D 1
            model.add(keras.layers.UpSampling2D());
            model.add(keras.layers.Conv2D(64, 3, 1, padding: "same", activation: activation));
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));

            //Conv2D 1
            model.add(keras.layers.Conv2D(32, 3, 1, padding: "same", activation: activation));
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));

            //Conv2D 1
            model.add(keras.layers.Conv2D(channels, 3, 1, padding: "same", activation: "tanh"));
            model.summary();//may do nothing since this is not a console app

            return model;
        }
        private Tensorflow.Keras.Engine.Model Make_Discriminator_model()
        {
            Activation activation = null;
            Tensor image = keras.Input(img_shape);
            //Tensor val2 = keras.Input(img_shape);

            //6 trainable layers max

            //Conv2D 1
            //Tensors x = keras.layers.Concatenate().Apply(image, val2);
            Tensors x = keras.layers.Conv2D(64, kernel_size: 5, strides: (2, 2), padding: "same", activation: activation).Apply(image);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            //x = keras.layers.Dropout(0.2f).Apply(x);

            //Conv2D 1
            x = keras.layers.Conv2D(128, 5, (2, 2), "same", activation: activation).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
            //x = keras.layers.Dropout(0.2f).Apply(x);

            //Conv2D 1
            x = keras.layers.Conv2D(256, 3, (2, 2), "same", activation: activation).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);

            //Conv2D 1
            x = keras.layers.Conv2D(512, 3, (2, 2), "same", activation: activation).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);

            //Conv2D 1
            x = keras.layers.Conv2D(1024, 3, (2, 2), "same", activation: activation).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);

            //Dense 1
            x = keras.layers.Flatten().Apply(x);
            x = keras.layers.Dense(2, activation: "sigmoid").Apply(x);

            Functional model = keras.Model(image, x);
            model.summary();

            return model;
        }
        private Tensor BinaryCrossentropy(Tensor x, Tensor y)
        {
            Tensor shape = tf.reduce_prod(tf.shape(x));
            Tensor count = tf.cast(shape, TF_DataType.TF_FLOAT);
            x = tf.clip_by_value(x, 1e-6f, 1.0f - 1e-6f);
            Tensor z = y * tf.log(x) + (1 - y) * tf.log(1 - x);
            Tensor result = -1.0f / count * tf.reduce_sum(z);
            return result;
        }
        private NDArray GenerateImage(Tensorflow.Keras.Engine.Model generator)
        {
            int r = 5;//output image rows
            int c = 5;//output image columns

            NDArray noise = np.random.normal(0, 1, new int[] { r * c, latent_dim });
            noise = noise.astype(np.float32);
            Tensor tensor_result = generator.predict(noise);
            return tensor_result.numpy();

        }

        //private void SaveImage(NDArray gen_imgs, int step)
        //{
        //    gen_imgs = gen_imgs * 0.5 + 0.5;
        //    int c = 5;
        //    long r = gen_imgs.shape[0] / c;
        //    NDArray nDArray = np.zeros((img_rows * r, img_cols * c), dtype: np.float32);
        //    for (int i = 0; i < r; i++)
        //    {
        //        for (int j = 0; j < c; j++)
        //        {
        //            var x = new Slice(i * img_rows, (i + 1) * img_cols);
        //            var y = new Slice(j * img_rows, (j + 1) * img_cols);
        //            var v = gen_imgs[i * r + j].reshape((img_rows, img_cols));
        //            nDArray[x, y] = v;
        //        }
        //    }

        //    NDArray t = nDArray.reshape((img_rows * r, img_cols * c)) * 255;
        //    //GrayToRGB(t.astype(np.uint8)).ToBitmap().Save(saveImgPath + "/image" + step + ".jpg");
        //}
        //private NDArray GrayToRGB(NDArray img2D)
        //{
        //    NDArray img4A = np.full_like(img2D, 255);
        //    NDArray img3D = np.expand_dims(img2D, 2);
        //    NDArray r = np.dstack(img3D, img3D, img3D, img4A);
        //    NDArray img4 = np.expand_dims(r, 0);
        //    return img4;
        //}
        //public void Test()
        //{
        //    var G = Make_Generator_model();
        //    G.load_weights(saveModelPath + "/Model_100_g.weights");
        //    PredictImage(G, 1);
        //}

        #endregion

        public void Train()//https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
        {
            try
            {
                float d_learnRate = 2e-4f;//0.0002 (?)
                float g_learnRate = 2e-4f;//0.0002 (?)

                OptimizerV2 d_optimizer = keras.optimizers.Adam(d_learnRate, 0.5f);
                OptimizerV2 g_optimizer = keras.optimizers.Adam(g_learnRate, 0.5f);

                Tensorflow.Keras.Engine.Model Generator = Make_Generator_model();//!Time: 124ms
                Tensorflow.Keras.Engine.Model Discriminator = Make_Discriminator_model();//!Time: 73ms

                LoadDataset();//moved here so the main thread does not block
                //Trainging loop
                for (currentEpoch = 0; currentEpoch <= epochAmount; currentEpoch++)
                {
                    stopWatch.Start();

                    var bth = datasetV2.take(1);
                    (Tensor, Tensor) batch = bth.First();//!Time: 26ms
                    images = batch.Item1;
                    if(conditional)
                        classValue = batch.Item2;
                    //NDArray randomIndexes = np.random.randint(0, (int)X_train.shape[0], size: batch_size);//Array of random indexes, length is batch size (?)
                    //NDArray realImgs = X_train[randomIndexes];//get array of images using indexs, of length of index array (?)

                    //Tensor g_loss, d_loss, d_loss_real, d_loss_fake;
                    using (GradientTape tape = tf.GradientTape(true))//!Time: 2ms
                    {
                        NDArray noise = np.random.normal(0, 1, new int[] { batch_size, generatorInputDims });//!Time: 20ms //Last 2 values are img count and latent dims (?)
                        noise = noise.astype(np.float32);

                        //NDArray secondDiscrimInput = new NDArray((batch_size, img_width, img_height, channels), TF_DataType.TF_FLOAT);

                        if (conditional)
                            for (int i = 0; i < batch_size; i++)//!Time: 2ms
                            {
                                noise[i, latent_dim] = (float)(int)classValue[i];//change this to set multiple with multiple inputs
                                //var a = secondDiscrimInput[i, 0, 0];
                                //secondDiscrimInput[i, 0, 0] = new float[] { 0f, 0f, (float)(int)classValue[i] };
                            }

                        fakeImgs = Generator.Apply(noise);//!Time: 41ms 
                        Tensor[] discrimOutFake = tf.split(Discriminator.Apply(fakeImgs), 2, 1);//!Time: 47ms 
                        Tensor[] discrimOutReal = tf.split(Discriminator.Apply(images), 2, 1);//!Time: 45ms 

                        //Computes the cross-entropy loss between true labels and predicted labels.
                        d_loss_real = BinaryCrossentropy(discrimOutReal[0], tf.ones_like(discrimOutReal[0])); //!Time: 6ms
                        Tensor d_loss_real_class = BinaryCrossentropy(discrimOutReal[1], classValue); //!Time: 6ms
                        //var test = tf.zeros_like(discrimOutFake[0]);
                        d_loss_fake = BinaryCrossentropy(discrimOutFake[0], tf.zeros_like(discrimOutFake[0])); //!Time: 7ms
                        Tensor d_loss_fake_class = BinaryCrossentropy(discrimOutFake[1], classValue); //!Time: 7ms
                        d_loss = d_loss_real + d_loss_fake; 

                        g_loss = BinaryCrossentropy(discrimOutFake[0], tf.ones_like(discrimOutFake[0])); //!Time: 7ms

                        //train Discriminator (?)
                        Tensors grad = tape.gradient(d_loss, Discriminator.trainable_variables);//!Time: 140 ms
                        d_optimizer.apply_gradients(zip(grad, Discriminator.trainable_variables.Select(x => x as ResourceVariable)));//!Time: 10ms

                        //train Generator (?)
                        grad = tape.gradient(g_loss, Generator.trainable_variables);//!Time: 95ms
                        g_optimizer.apply_gradients(zip(grad, Generator.trainable_variables.Select(x => x as ResourceVariable))); //!Time: 8ms
                    }

                    //if (currentEpoch % save_every == 0 && currentEpoch != 0)
                    //{
                    //        PredictImage(Generator, currentEpoch);
                    //}

                    //if (currentEpoch % 100 == 0)
                    //{
                    //    Generator.save_weights(saveModelPath + "/Model_" + currentEpoch + "_g.weights");
                    //    Discriminator.save_weights(saveModelPath + "/Model_" + currentEpoch + "_d.weights");
                    //}

                    //needed or else it will OOM before garbage collection starts
                    GC.Collect();//!Time: 9ms 

                    stopWatch.Stop();
                    lastStopwatchTime = stopWatch.Elapsed;
                    stopWatch.Reset();
                }
            }
            catch(Exception e)
            {
                OutputText = e.Message;
            }
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();

            if (!StartedTraining)
            {
                outputImageFake = new Texture2D(Graphics.GraphicsDevice, img_height, img_width);
                outputImage = new Texture2D(Graphics.GraphicsDevice, img_height, img_width);
                //Train();
                Task.Factory.StartNew(() => { Train(); }, TaskCreationOptions.LongRunning);//IMPORTANT: all errors on this thread will not show up in vs and will sliently stop training
                StartedTraining = true;
            }

            if (currentEpoch != lastDrawEpoch)
            {
                ThreadPool.QueueUserWorkItem(state => BuildOutputText());

                ThreadPool.QueueUserWorkItem(state => BuildFakeImageData());

                ThreadPool.QueueUserWorkItem(state => BuildRealImageData());

                lastDrawEpoch = currentEpoch;
            }

            base.Update(gameTime);
        }

        public void BuildOutputText()
        {
            //text generation
            try
            {
                float s_d_loss_real = tf.reduce_mean(d_loss_real).numpy();
                float s_d_loss_fake = tf.reduce_mean(d_loss_fake).numpy();
                float s_d_loss = tf.reduce_mean(d_loss).numpy();
                float s_g_loss = tf.reduce_mean(g_loss).numpy();
                OutputText = $"Epoch: {currentEpoch}   d_loss: {s_d_loss} (Real: {s_d_loss_real} + Fake: {s_d_loss_fake}) g_loss: {s_g_loss}";
            }
            catch (Exception E)
            {

            }
        }

        public void BuildFakeImageData()
        {
            float[] fakeImageData = fakeImgs[0][0].ToArray<float>();

            int size = img_width * img_height;
            Color[] dataCache = new Color[size];


            for (int pixel = 0; pixel < size; pixel++)
            {
                //the function applies the color according to the specified pixel
                if(channels == 3)
                    dataCache[pixel] = new Color(Unscale(fakeImageData[(pixel * 3)]), Unscale(fakeImageData[(pixel * 3) + 1]), Unscale(fakeImageData[(pixel * 3) + 2]));//rgb
                else
                {
                    float val = Unscale(fakeImageData[(pixel)]);//grayscale
                    dataCache[pixel] = new Color(val, val, val);//grayscale
                }
            }
            outputImageFake.SetData(dataCache);
        }

        public void BuildRealImageData()
        {
            float[] imageData = images[0].ToArray<float>();

            int size = img_width * img_height;
            Color[] dataCache = new Color[size];

            for (int pixel = 0; pixel < size; pixel++)
            {
                if (channels == 3)
                    dataCache[pixel] = new Color(Unscale(imageData[(pixel * 3)]), Unscale(imageData[(pixel * 3) + 1]), Unscale(imageData[(pixel * 3) + 2]));
                else
                {
                    float val = Unscale(imageData[(pixel)]);//grayscale
                    dataCache[pixel] = new Color(val, val, val);//grayscale
                }
            }

            outputImage.SetData(dataCache);
        }

        public int Unscale(float val) => 
            (int)((val + 1f) * 127.5f);

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.CornflowerBlue);

            SpriteBatch.Begin(default, default, SamplerState.PointClamp);

            SpriteBatch.Draw(debug, windowSize / 2 - new Vector2(8, 8), null, Color.White, 0, default, 8f, default, default);
            if (outputImageFake != null)
                SpriteBatch.Draw(outputImageFake, windowSize / 2, null, Color.White, 0, default, 4f, default, default);
            if (outputImage != null)
                SpriteBatch.Draw(outputImage, new Vector2(windowSize.X / 3, windowSize.Y / 2), null, Color.White, 0, default, 4f, default, default);

            SpriteBatch.DrawString(font_Arial, "Epoch time:  " + string.Format("{0:00}.{1:00}", lastStopwatchTime.Seconds, lastStopwatchTime.Milliseconds / 10) + " seconds.", new Vector2(20, 20), Color.White * 0.9f);
            for (int i = 0; i < DisplayText.Count; i++)
                SpriteBatch.DrawString(font_Arial, DisplayText[i], new Vector2(20, 50 + (20 * i)), Color.White * 0.8f);
            
            SpriteBatch.End();

            base.Draw(gameTime);
        }
    }
}
