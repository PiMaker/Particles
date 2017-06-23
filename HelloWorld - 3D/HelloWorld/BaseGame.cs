// File: BaseGame.cs
// Created: 23.06.2017
// 
// See <summary> tags for more information.

using System;
using System.IO;
using System.Text.RegularExpressions;
using ManagedCuda;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using MonoGame.Extended;

namespace HelloWorld
{
    public class BaseGame : Game
    {
        private const bool FIXED_TIMESTEP = false;

        private const int PARTICLES = 4000;
        private const int PARTICLE_SIZE = 7;

        private const int THREADS_PER_BLOCK = 1024;

        private static readonly Random Random = new Random();
        private readonly FrameCounter frameCounter = new FrameCounter();
        private readonly GraphicsDeviceManager graphics;
        private readonly float[] result = new float[BaseGame.PARTICLES * BaseGame.PARTICLE_SIZE];
        private readonly FrameCounter updateCounter = new FrameCounter();

        private float angle;
        private VertexPositionColor[][] cube;
        private BasicEffect effect;

        private SpriteFont font;
        private CudaKernel kernel;
        private TimeSpan lastTickTime = TimeSpan.Zero;
        private float maxInfoTextWidth;
        private CudaDeviceVariable<float> particleMemory1;
        private CudaDeviceVariable<float> particleMemory2;
        private SpriteBatch spriteBatch;
        private Matrix view, project;

        public BaseGame()
        {
            this.graphics = new GraphicsDeviceManager(this);
            this.Content.RootDirectory = "Content";
        }

        protected override void Initialize()
        {
            this.graphics.SynchronizeWithVerticalRetrace = BaseGame.FIXED_TIMESTEP;
            this.IsFixedTimeStep = BaseGame.FIXED_TIMESTEP;
            this.graphics.PreferredBackBufferWidth = 1600;
            this.graphics.PreferredBackBufferHeight = 900;
            this.graphics.ApplyChanges();

            this.InitKernels();
            this.InitParticles();

            // Initialize 3D stuff
            this.view = Matrix.CreateLookAt(new Vector3(0f, 0f, -5f), Vector3.Zero, Vector3.Up);
            this.project = Matrix.CreatePerspectiveFieldOfView(MathHelper.PiOver2,
                this.graphics.PreferredBackBufferWidth / (float)this.graphics.PreferredBackBufferHeight, 1, 100);

            base.Initialize();
        }

        protected override void LoadContent()
        {
            // Create a new SpriteBatch, which can be used to draw textures.
            this.spriteBatch = new SpriteBatch(this.GraphicsDevice);

            this.font = this.Content.Load<SpriteFont>("font");
            this.effect = new BasicEffect(this.GraphicsDevice);

            // Load cubes
            this.cube = new VertexPositionColor[256][];
            for (var i = 0; i < 256; i++)
            {
                this.cube[i] = BaseGame.BuildCube(new Color((byte) 255, (byte) 0, (byte) 0, (byte) i));
            }
        }

        protected override void UnloadContent()
        {
        }

        protected override void Update(GameTime gameTime)
        {
            this.updateCounter.Update((float) gameTime.ElapsedGameTime.TotalSeconds);

            if (Keyboard.GetState().IsKeyDown(Keys.Escape))
            {
                this.Exit();
            }

            var now = DateTime.Now;

            // Update routine

            this.angle += 0.0125f;

            // Prepare steps
            var delta = (float) gameTime.ElapsedGameTime.TotalMilliseconds;
            
            // Call kernel
            this.kernel.Run(this.particleMemory1.DevicePointer, this.particleMemory2.DevicePointer,
                BaseGame.PARTICLES, delta);

            // Retrieve result
            this.particleMemory2.CopyToHost(this.result);

            // Swap memory pointers
            var temp = this.particleMemory2;
            this.particleMemory2 = this.particleMemory1;
            this.particleMemory1 = temp;

            this.lastTickTime = DateTime.Now - now;
            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            this.frameCounter.Update((float) gameTime.ElapsedGameTime.TotalSeconds);

            this.GraphicsDevice.Clear(Color.White);

            this.effect.View = this.view;
            this.effect.Projection = this.project;
            this.effect.VertexColorEnabled = true;

            this.effect.World = Matrix.Identity;
            this.effect.CurrentTechnique.Passes[0].Apply();

            var count = 0;
            
            // Draw particles
            for (var i = 0; i < BaseGame.PARTICLES; i++)
            {
                var pi = i * BaseGame.PARTICLE_SIZE;
                if (this.result[pi] > 1/255f)
                {
                    var pos = new Vector3(this.result[pi + 1], this.result[pi + 2], this.result[pi + 3]);
                    this.DrawCube(this.cube[(int) (this.result[pi] * 255)], pos, 0.002f);
                    count++;
                }
            }

            this.spriteBatch.Begin();

            // Draw Infotext
            var text =
                $"Tick: {Math.Round(this.lastTickTime.TotalMilliseconds, 2)} ms\nUPS: {Math.Round(this.updateCounter.AverageFramesPerSecond, 2)}\nFPS: {Math.Round(this.frameCounter.AverageFramesPerSecond, 2)}\nParticles: {count}/{BaseGame.PARTICLES}";
            var size = this.font.MeasureString(text);
            size.X = this.maxInfoTextWidth = Math.Max(this.maxInfoTextWidth, size.X);
            var position = Vector2.One * 5;

            this.spriteBatch.FillRectangle(Vector2.Zero, size + position * 2, Color.LightGray * 0.7f);
            this.spriteBatch.DrawString(this.font, text, position, Color.Green);

            this.spriteBatch.End();
            base.Draw(gameTime);
        }

        private void InitKernels()
        {
            var kernelName = File.ReadAllText("kernel.ptx");
            var match = Regex.Match(kernelName, @"\/\/ \.globl\s(.*?)\r?\n");
            kernelName = match.Groups[1].Value;

            var cntxt = new CudaContext();
            var cumodule = cntxt.LoadModule(@"kernel.ptx");
            this.kernel = new CudaKernel(kernelName, cumodule, cntxt)
            {
                BlockDimensions = BaseGame.THREADS_PER_BLOCK,
                GridDimensions = BaseGame.PARTICLES / BaseGame.THREADS_PER_BLOCK + 1
            };
        }

        private void InitParticles()
        {
            var particles = new float[BaseGame.PARTICLES * BaseGame.PARTICLE_SIZE];

            for (var i = 0; i < BaseGame.PARTICLES; i++)
            {
                var pi = i * BaseGame.PARTICLE_SIZE;
                particles[pi] = BaseGame.Random.NextSingle(0f, 1f);

                // Random position (X, Y, Z)
                particles[pi + 1] =
                    particles[pi + 4] = BaseGame.Random.NextSingle(-0.1f, 0.1f);
                particles[pi + 4] += BaseGame.Random.NextSingle(-0.0005f, 0.0005f);
                
                particles[pi + 2] = particles[pi + 5] = -3.5f + BaseGame.Random.NextSingle(-0.1f, 0.1f);
                particles[pi + 5] -= 0.008f;
                
                particles[pi + 3] =
                    particles[pi + 6] = BaseGame.Random.NextSingle(-0.1f, 0.1f);
                particles[pi + 6] += BaseGame.Random.NextSingle(-0.0005f, 0.0005f);
            }

            // Initialize memory1 and memory2 with actual particles
            this.particleMemory1 = new CudaDeviceVariable<float>(BaseGame.PARTICLES * BaseGame.PARTICLE_SIZE);
            this.particleMemory1.CopyToDevice(particles);
            this.particleMemory2 = new CudaDeviceVariable<float>(BaseGame.PARTICLES * BaseGame.PARTICLE_SIZE);
            this.particleMemory2.CopyToDevice(particles);
        }

        private void DrawCube(VertexPositionColor[] model, Vector3 position, float size)
        {
            this.effect.World = Matrix.CreateScale(size) *
                                Matrix.CreateTranslation(position);
            this.effect.CurrentTechnique.Passes[0].Apply();
            this.GraphicsDevice.DrawUserPrimitives(PrimitiveType.TriangleList, model, 0, model.Length / 3);
        }

        private static VertexPositionColor[] BuildCube(Color cubeColor)
        {
            var topLeftFront = new Vector3(-1.0f, 1.0f, -1.0f);
            var topRightFront = new Vector3(1.0f, 1.0f, -1.0f);
            var topLeftBack = new Vector3(-1.0f, 1.0f, 1.0f);
            var topRightBack = new Vector3(1.0f, 1.0f, 1.0f);

            var botLeftFront = new Vector3(-1.0f, -1.0f, -1.0f);
            var botRightFront = new Vector3(1.0f, -1.0f, -1.0f);
            var botLeftBack = new Vector3(-1.0f, -1.0f, 1.0f);
            var botRightBack = new Vector3(1.0f, -1.0f, 1.0f);

            var arr = new VertexPositionColor[36];

            arr[0] = new VertexPositionColor(topLeftFront, cubeColor);
            arr[1] = new VertexPositionColor(botLeftFront, cubeColor);
            arr[2] = new VertexPositionColor(topRightFront, cubeColor);
            arr[3] = new VertexPositionColor(botLeftFront, cubeColor);
            arr[4] = new VertexPositionColor(botRightFront, cubeColor);
            arr[5] = new VertexPositionColor(topRightFront, cubeColor);

            arr[6] = new VertexPositionColor(topLeftBack, cubeColor);
            arr[7] = new VertexPositionColor(topRightBack, cubeColor);
            arr[8] = new VertexPositionColor(botLeftBack, cubeColor);
            arr[9] = new VertexPositionColor(botLeftBack, cubeColor);
            arr[10] = new VertexPositionColor(topRightBack, cubeColor);
            arr[11] = new VertexPositionColor(botRightBack, cubeColor);

            arr[12] = new VertexPositionColor(topLeftFront, cubeColor);
            arr[13] = new VertexPositionColor(topRightBack, cubeColor);
            arr[14] = new VertexPositionColor(topLeftBack, cubeColor);
            arr[15] = new VertexPositionColor(topLeftFront, cubeColor);
            arr[16] = new VertexPositionColor(topRightFront, cubeColor);
            arr[17] = new VertexPositionColor(topRightBack, cubeColor);

            arr[18] = new VertexPositionColor(botLeftFront, cubeColor);
            arr[19] = new VertexPositionColor(botLeftBack, cubeColor);
            arr[20] = new VertexPositionColor(botRightBack, cubeColor);
            arr[21] = new VertexPositionColor(botLeftFront, cubeColor);
            arr[22] = new VertexPositionColor(botRightBack, cubeColor);
            arr[23] = new VertexPositionColor(botRightFront, cubeColor);

            arr[24] = new VertexPositionColor(topLeftFront, cubeColor);
            arr[25] = new VertexPositionColor(botLeftBack, cubeColor);
            arr[26] = new VertexPositionColor(botLeftFront, cubeColor);
            arr[27] = new VertexPositionColor(topLeftBack, cubeColor);
            arr[28] = new VertexPositionColor(botLeftBack, cubeColor);
            arr[29] = new VertexPositionColor(topLeftFront, cubeColor);

            arr[30] = new VertexPositionColor(topRightFront, cubeColor);
            arr[31] = new VertexPositionColor(botRightFront, cubeColor);
            arr[32] = new VertexPositionColor(botRightBack, cubeColor);
            arr[33] = new VertexPositionColor(topRightBack, cubeColor);
            arr[34] = new VertexPositionColor(topRightFront, cubeColor);
            arr[35] = new VertexPositionColor(botRightBack, cubeColor);

            return arr;
        }
    }
}

/*

p - Memory layout:

i = particle index * 11
i = opacity, active if > 0
i + 1 = Position X
i + 2 = Position Y
i + 3 = Position Z
i + 4 = Prev. Position X
i + 5 = Prev. Position Y
i + 6 = Prev. Position Z

*/