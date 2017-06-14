// File: BaseGame.cs
// Created: 14.06.2017
// 
// See <summary> tags for more information.

using System;
using System.Linq;
using ManagedCuda;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

namespace HelloWorld
{
    public class BaseGame : Game
    {
        private const int VECTOR_SIZE = 5120;
        private const int THREADS_PER_BLOCK = 1024;

        private static CudaKernel addTwoVectorWithCuda;

        private static readonly Func<int[], int[], int, int[]> addVectors = (a, b, size) =>
        {
            // init parameters
            CudaDeviceVariable<int> vectorHostA = a;
            CudaDeviceVariable<int> vectorHostB = b;
            var vectorHostOut = new CudaDeviceVariable<int>(size);
            // run cuda method
            BaseGame.addTwoVectorWithCuda.Run(vectorHostA.DevicePointer, vectorHostB.DevicePointer,
                vectorHostOut.DevicePointer, size);
            // copy return to host
            var output = new int[size];
            vectorHostOut.CopyToHost(output);
            return output;
        };

        private GraphicsDeviceManager graphics;
        private SpriteBatch spriteBatch;

        public BaseGame()
        {
            this.graphics = new GraphicsDeviceManager(this);
            this.Content.RootDirectory = "Content";
        }

        protected override void Initialize()
        {
            base.Initialize();
            
            // TEST
            Test();
        }

        protected override void LoadContent()
        {
            // Create a new SpriteBatch, which can be used to draw textures.
            this.spriteBatch = new SpriteBatch(this.GraphicsDevice);
        }

        protected override void UnloadContent()
        {
        }

        protected override void Update(GameTime gameTime)
        {
            if (Keyboard.GetState().IsKeyDown(Keys.Escape))
            {
                this.Exit();
            }

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            this.GraphicsDevice.Clear(Color.CornflowerBlue);

            base.Draw(gameTime);
        }

        private static void InitKernels()
        {
            var cntxt = new CudaContext();
            var cumodule =
                cntxt.LoadModule(
                    @"kernel.ptx");
            BaseGame.addTwoVectorWithCuda = new CudaKernel("_Z6kernelPiS_S_i", cumodule, cntxt);
            BaseGame.addTwoVectorWithCuda.BlockDimensions = BaseGame.THREADS_PER_BLOCK;
            BaseGame.addTwoVectorWithCuda.GridDimensions = BaseGame.VECTOR_SIZE / BaseGame.THREADS_PER_BLOCK + 1;
        }

        private static void Test()
        {
            BaseGame.InitKernels();
            var vectorA = Enumerable.Range(1, BaseGame.VECTOR_SIZE).ToArray();
            var vectorB = Enumerable.Range(1, BaseGame.VECTOR_SIZE).ToArray();
            var vector = BaseGame.addVectors(vectorA, vectorB, BaseGame.VECTOR_SIZE);
            for (var i = 0; i < BaseGame.VECTOR_SIZE; i++)
            {
                Console.WriteLine("{0}+{1}={2}", vectorA[i], vectorB[i], vector[i]);
            }
        }
    }
}