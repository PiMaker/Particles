using System.Collections.Generic;
using System.Linq;

namespace HelloWorld
{
    // From: https://stackoverflow.com/a/20679895
    public class FrameCounter
    {
        public const int MAXIMUM_SAMPLES = 100;

        private readonly Queue<float> sampleBuffer = new Queue<float>();
        public long TotalFrames { get; private set; }
        public float TotalSeconds { get; private set; }
        public float AverageFramesPerSecond { get; private set; }
        public float CurrentFramesPerSecond { get; private set; }

        public bool Update(float deltaTime)
        {
            this.CurrentFramesPerSecond = 1.0f / deltaTime;

            this.sampleBuffer.Enqueue(this.CurrentFramesPerSecond);

            if (this.sampleBuffer.Count > FrameCounter.MAXIMUM_SAMPLES)
            {
                this.sampleBuffer.Dequeue();
                this.AverageFramesPerSecond = this.sampleBuffer.Average(i => i);
            }
            else
            {
                this.AverageFramesPerSecond = this.CurrentFramesPerSecond;
            }

            this.TotalFrames++;
            this.TotalSeconds += deltaTime;
            return true;
        }
    }
}