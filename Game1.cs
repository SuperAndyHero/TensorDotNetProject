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

namespace TensorDotNetProject
{
    public class Game1 : Game
    {
        //TODO: compare with base tutorial

        #region variables

        GraphicsDeviceManager Graphics;
        SpriteBatch SpriteBatch;


        float LeakyReLU_alpha = 0.2f;


        int epochAmount = 1000;
        //int epochs = 2000; // Better effect, but longer time
        int batch_size = 64;

        //int save_every = 10;
        //int display_every = 5;

        string saveImgPath = "D:/Pictures/aGanTest/imgs";
        string saveModelPath = "D:/Pictures/aGanTest/models";

        int latent_dim = 100;

        int img_rows = 28;
        int img_cols = 28;
        int channels = 1;


        Shape img_shape;

        DatasetPass dataset;

        SpriteFont font_Arial;
        Texture2D debug;

        Vector2 windowSize;

        //string deviceName = "";
        TimeSpan lastStopwatchTime = new TimeSpan(1);
        string outputText = "";
        Texture2D outputImage;

        Tensor g_loss, d_loss, d_loss_real, d_loss_fake;
        Tensors fakeImgs;

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
            dataset = keras.datasets.mnist.load_data();

            img_shape = (img_rows, img_cols, channels);

            if (img_cols % 4 != 0 || img_rows % 4 != 0)
            {
                throw new Exception("The width and height of the image must be a multiple of 4");
            }

            System.IO.Directory.CreateDirectory(saveImgPath);
            System.IO.Directory.CreateDirectory(saveModelPath);
        }

        #endregion


        #region methods

        private Tensorflow.Keras.Engine.Model Make_Generator_model()
        {
            Activation activation = null;

            Sequential model = keras.Sequential();
            model.add(keras.layers.Dense((img_rows / 4) * (img_cols / 4) * 256, activation: activation, input_shape: 100));//2 bracket sets added around (int / 4) operations
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));

            model.add(keras.layers.Reshape((7, 7, 256)));//the first two seem to be the input divided by 4 (?)

            model.add(keras.layers.UpSampling2D());
            model.add(keras.layers.Conv2D(128, 3, 1, padding: "same", activation: activation));
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));

            model.add(keras.layers.UpSampling2D());
            model.add(keras.layers.Conv2D(64, 3, 1, padding: "same", activation: activation));
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));

            model.add(keras.layers.Conv2D(32, 3, 1, padding: "same", activation: activation));
            model.add(keras.layers.BatchNormalization(momentum: 0.8f));
            model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));

            model.add(keras.layers.Conv2D(1, 3, 1, padding: "same", activation: "tanh"));
            model.summary();//may do nothing since this is not a console app

            return model;
        }
        private Tensorflow.Keras.Engine.Model Make_Discriminator_model()
        {
            Activation activation = null;
            Tensor image = keras.Input(img_shape);

            Tensors x = keras.layers.Conv2D(128, kernel_size: 3, strides: (2, 2), padding: "same", activation: activation).Apply(image);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);

            x = keras.layers.Conv2D(256, 3, (2, 2), "same", activation: activation).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);

            x = keras.layers.Conv2D(512, 3, (2, 2), "same", activation: activation).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);

            x = keras.layers.Conv2D(1024, 3, (2, 2), "same", activation: activation).Apply(x);
            x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
            x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);

            x = keras.layers.Flatten().Apply(x);
            x = keras.layers.Dense(1, activation: "sigmoid").Apply(x);

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

        public void Train()
        {
            NDArray X_train = dataset.Train.Item1;
            X_train = X_train / 127.5 - 1; //Normalize the images to [-1, 1] (?)
            X_train = np.expand_dims(X_train, 3);
            X_train = X_train.astype(np.float32);

            Tensorflow.Keras.Engine.Model Generator = Make_Generator_model();
            Tensorflow.Keras.Engine.Model Discriminator = Make_Discriminator_model();

            float d_learnRate = 2e-4f;//0.0002 (?)
            float g_learnRate = 2e-4f;//0.0002 (?)

            OptimizerV2 d_optimizer = keras.optimizers.Adam(d_learnRate, 0.5f);
            OptimizerV2 g_optimizer = keras.optimizers.Adam(g_learnRate, 0.5f);

            for (currentEpoch = 0; currentEpoch <= epochAmount; currentEpoch++)
            {
                stopWatch.Start();
                NDArray randomIndexes = np.random.randint(0, (int)X_train.shape[0], size: batch_size);//Array of random indexes, length is batch size (?)
                NDArray realImgs = X_train[randomIndexes];//get array of images using indexs, of length of index array (?)

                //Tensor g_loss, d_loss, d_loss_real, d_loss_fake;
                using (GradientTape tape = tf.GradientTape(true))
                {
                    NDArray noise = np.random.normal(0, 1, new int[] { batch_size, 100 });//Last 2 values are img count and latent dims (?)
                    noise = noise.astype(np.float32);

                    fakeImgs = Generator.Apply(noise);
                    Tensors discrimOutFake = Discriminator.Apply(fakeImgs);
                    Tensors discrimOutReal = Discriminator.Apply(realImgs);

                    d_loss_real = BinaryCrossentropy(discrimOutReal, tf.ones_like(discrimOutReal));
                    d_loss_fake = BinaryCrossentropy(discrimOutFake, tf.zeros_like(discrimOutFake));

                    g_loss = BinaryCrossentropy(discrimOutFake, tf.ones_like(discrimOutFake));
                    d_loss = d_loss_real + d_loss_fake;

                    //train Discriminator (?)
                    Tensors grad = tape.gradient(d_loss, Discriminator.trainable_variables);
                    d_optimizer.apply_gradients(zip(grad, Discriminator.trainable_variables.Select(x => x as ResourceVariable)));

                    //train Generator (?)
                    grad = tape.gradient(g_loss, Generator.trainable_variables);
                    g_optimizer.apply_gradients(zip(grad, Generator.trainable_variables.Select(x => x as ResourceVariable)));
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

                GC.Collect();//needed or else it will OOM before garbage collection starts

                stopWatch.Stop();
                lastStopwatchTime = stopWatch.Elapsed;
                stopWatch.Reset();
            }

        }

        #endregion


        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();

            if (!StartedTraining)
            {
                outputImage = new Texture2D(Graphics.GraphicsDevice, img_cols, img_rows);
                Task.Factory.StartNew(() => { Train(); }, TaskCreationOptions.LongRunning);//IMPORTANT: all errors on this thread will not show up in vs and will sliently stop training
                StartedTraining = true;
            }

            if (currentEpoch != lastDrawEpoch)
            {
                //text generation
                float s_d_loss_real = tf.reduce_mean(d_loss_real).numpy();
                float s_d_loss_fake = tf.reduce_mean(d_loss_fake).numpy();
                float s_d_loss = tf.reduce_mean(d_loss).numpy();
                float s_g_loss = tf.reduce_mean(g_loss).numpy();
                outputText = $"Epoch: {currentEpoch} \nd_loss: {s_d_loss} (Real: {s_d_loss_real} + Fake: {s_d_loss_fake}) \ng_loss: {s_g_loss}";

                //image generation
                float[] imageData = fakeImgs[0][0].ToArray<float>();
                int size = img_rows * img_cols;
                Color[] data = new Color[size];
                for (int pixel = 0; pixel < size; pixel++)
                {
                    //the function applies the color according to the specified pixel
                    //data[pixel] = new Color((imageData[pixel + (size * 2)]), (imageData[pixel + (size)]), (imageData[pixel]));//rgb
                    float val = imageData[pixel] * 0.5f + 0.5f;
                    data[pixel] = new Color(val, val, val);
                }
                outputImage.SetData(data);

                lastDrawEpoch = currentEpoch;
            }

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.CornflowerBlue);

            SpriteBatch.Begin(default, default, SamplerState.PointClamp);

            SpriteBatch.Draw(debug, windowSize / 2 - new Vector2(8, 8), null, Color.White, 0, default, 8f, default, default);
            if (outputImage != null)
                SpriteBatch.Draw(outputImage, windowSize / 2, null, Color.White, 0, default, 4f, default, default);

            SpriteBatch.DrawString(font_Arial, "Epoch time:  " + string.Format("{0:00}.{1:00}", lastStopwatchTime.Seconds, lastStopwatchTime.Milliseconds / 10) + " seconds.", new Vector2(20, 20), Color.White * 0.9f);
            SpriteBatch.DrawString(font_Arial, outputText, new Vector2(20, 50), Color.White * 0.8f);
            
            SpriteBatch.End();

            base.Draw(gameTime);
        }
    }
}
