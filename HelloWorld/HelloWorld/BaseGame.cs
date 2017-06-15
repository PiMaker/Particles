// File: BaseGame.cs
// Created: 14.06.2017
// 
// See <summary> tags for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using ManagedCuda;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using MonoGame.Extended;
using MonoGame.Extended.Shapes;

namespace HelloWorld
{
    public class BaseGame : Game
    {
        private const bool FIXED_TIMESTEP = false;

        private const int PARTICLES = 4000;
        private const int PARTICLE_SIZE = 10;
        private const float RADIUS = 4;

        private const int THREADS_PER_BLOCK = 1024;
        private const int ITERATIONS = 1;

        private static readonly Random random = new Random();
        private static readonly Polygon circle = BaseGame.CreateCircle(BaseGame.RADIUS, 6);
        private readonly FrameCounter frameCounter = new FrameCounter();
        private readonly GraphicsDeviceManager graphics;
        private readonly float[] result = new float[BaseGame.PARTICLES * BaseGame.PARTICLE_SIZE];
        private readonly FrameCounter updateCounter = new FrameCounter();

        private SpriteFont font;
        private CudaKernel kernel;
        private TimeSpan lastTickTime = TimeSpan.Zero;
        private float maxInfoTextWidth;
        private CudaDeviceVariable<float> particleMemory1;
        private CudaDeviceVariable<float> particleMemory2;
        private SpriteBatch spriteBatch;
        private CudaDeviceVariable<float> area;

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

            // Coordinates: x, y, x_far, y_far
            this.area = new CudaDeviceVariable<float>(4);
            this.area.CopyToDevice(new float[] { 0, 0, this.graphics.PreferredBackBufferWidth, this.graphics.PreferredBackBufferHeight });

            base.Initialize();
        }

        protected override void LoadContent()
        {
            // Create a new SpriteBatch, which can be used to draw textures.
            this.spriteBatch = new SpriteBatch(this.GraphicsDevice);

            this.font = this.Content.Load<SpriteFont>("font");
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

            // Prepare steps
            var delta = (float)gameTime.ElapsedGameTime.TotalMilliseconds/BaseGame.ITERATIONS;

            for (var i = 0; i < BaseGame.ITERATIONS; i++)
            {
                // Call kernel
                this.kernel.Run(this.particleMemory1.DevicePointer, this.particleMemory2.DevicePointer, BaseGame.PARTICLES, delta, this.area.DevicePointer);

                // Swap memory pointers
                var temp = this.particleMemory2;
                this.particleMemory2 = this.particleMemory1;
                this.particleMemory1 = temp;
            }

            // Retrieve result
            this.particleMemory1.CopyToHost(this.result);

            this.lastTickTime = DateTime.Now - now;
            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            this.frameCounter.Update((float) gameTime.ElapsedGameTime.TotalSeconds);

            this.GraphicsDevice.Clear(Color.White);
            this.spriteBatch.Begin();

            // Draw center point
            this.spriteBatch.DrawCircle(new Vector2(this.graphics.PreferredBackBufferWidth/2f, this.graphics.PreferredBackBufferHeight/2f), 10, 32, Color.Black, 3f);

            // Draw particles
            for (var i = 0; i < BaseGame.PARTICLES; i++)
            {
                var pi = i * BaseGame.PARTICLE_SIZE;
                if (this.result[pi] > 0)
                {
                    this.spriteBatch.DrawPolygon(new Vector2(this.result[pi + 1], this.result[pi + 2]), BaseGame.circle,
                        BaseGame.FromFloatArray(this.result, pi + 6));
                }
            }

            // Draw Infotext
            var text =
                $"Tick: {Math.Round(this.lastTickTime.TotalMilliseconds, 2)} ms\nUPS: {Math.Round(this.updateCounter.AverageFramesPerSecond, 2)}\nFPS: {Math.Round(this.frameCounter.AverageFramesPerSecond, 2)}\nParticles: {BaseGame.PARTICLES}";
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
                particles[pi] = 1;

                // Random position
                particles[pi + 1] =
                    particles[pi + 3] = (BaseGame.random.NextSingle(0.05f, 0.95f)) * this.graphics.PreferredBackBufferWidth;
                particles[pi + 2] =
                    particles[pi + 4] = BaseGame.random.NextSingle(0.05f, 0.95f) * this.graphics.PreferredBackBufferHeight;

                // Radius
                particles[pi + 5] = BaseGame.RADIUS;

                // Color
                particles[pi + 6] = 1;
                particles[pi + 9] = 1;
            }

            // Initialize memory1 and memory2 with actual particles
            this.particleMemory1 = new CudaDeviceVariable<float>(BaseGame.PARTICLES * BaseGame.PARTICLE_SIZE);
            this.particleMemory1.CopyToDevice(particles);
            this.particleMemory2 = new CudaDeviceVariable<float>(BaseGame.PARTICLES * BaseGame.PARTICLE_SIZE);
            this.particleMemory2.CopyToDevice(particles);
        }

        // From Monogame.Extended
        private static Polygon CreateCircle(double radius, int sides)
        {
            var vectors = new List<Vector2>();

            const double MAX = 2.0 * Math.PI;
            var step = MAX / sides;

            for (var theta = 0.0; theta < MAX; theta += step)
            {
                vectors.Add(new Vector2((float) (radius * Math.Cos(theta)), (float) (radius * Math.Sin(theta))));
            }

            // then add the first vector again so it's a complete loop
            vectors.Add(new Vector2((float) (radius * Math.Cos(0)), (float) (radius * Math.Sin(0))));
            return new Polygon(vectors);
        }

        private static Color FromFloatArray(float[] arr, int i)
        {
            return new Color(arr[i], arr[i + 1], arr[i + 2], arr[i + 3]);
        }
    }
}

/*

p - Memory layout:

i = particle index * 10
i = active if > 0
i + 1 = Position X
i + 2 = Position Y
i + 3 = Prev. Position X
i + 4 = Prev. Position Y
i + 5 = Radius
i + 6/7/8/9 = Color (RGBA)

*/