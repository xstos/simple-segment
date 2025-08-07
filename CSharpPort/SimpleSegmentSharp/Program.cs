using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

public class Segment
{
    public double X0 { get; set; }
    public double Y0 { get; set; }
    public double X1 { get; set; }
    public double Y1 { get; set; }

    public Segment(double x0, double y0, double x1, double y1)
    {
        X0 = x0;
        Y0 = y0;
        X1 = x1;
        Y1 = y1;
    }
}

public class SimpleSegment
{
    public static void Main(string[] args)
    {
        Example();
    }
    public static void Example()
    {
        var fileLines = new List<string>
        {
            "0.000\t-0.165\t-0.325\n",
            "0.008\t-0.155\t-0.325\n",
            "0.016\t-0.195\t-0.305\n",
            "0.023\t-0.205\t-0.305\n",
            "0.031\t-0.185\t-0.295\n",
            "0.039\t-0.155\t-0.265\n",
            "0.047\t-0.135\t-0.235\n",
            "0.055\t-0.095\t-0.185\n",
            "0.062\t-0.075\t-0.135\n",
            "0.070\t-0.065\t-0.095\n",
            "0.078\t-0.065\t-0.055\n",
            "0.086\t-0.125\t-0.015\n",
            "0.094\t-0.125\t 0.005\n",
            "0.102\t-0.125\t-0.045\n",
            "0.109\t-0.115\t-0.015\n",
            "0.117\t-0.125\t-0.005\n",
            "0.125\t-0.165\t-0.015\n"
        };

        var data = fileLines.Select(line => double.Parse(line.Split('\t')[2].Trim())).ToList();
        double maxError = 0.005;

        // Sliding window with regression
        var segments1 = SlidingWindowSegment(data, Regression, SumSquaredError, maxError);
        
        // Bottom-up with regression
        var segments2 = BottomUpSegment(data, Regression, SumSquaredError, maxError);
        
        // Top-down with regression
        var segments3 = TopDownSegment(data, Regression, SumSquaredError, maxError);
        
        // Sliding window with simple interpolation
        var segments4 = SlidingWindowSegment(data, Interpolate, SumSquaredError, maxError);
        
        // Bottom-up with simple interpolation
        var segments5 = BottomUpSegment(data, Interpolate, SumSquaredError, maxError);
        
        // Top-down with simple interpolation
        var segments6 = TopDownSegment(data, Interpolate, SumSquaredError, maxError);
    }
    public static Tuple<double[], double> LeastSquaresLineFit(List<double> sequence, Tuple<int, int> seqRange)
    {
        int start = seqRange.Item1;
        int end = seqRange.Item2;
        int n = end - start + 1;
        
        var x = Vector<double>.Build.Dense(n, i => start + i);
        var y = Vector<double>.Build.Dense(sequence.Skip(start).Take(n).ToArray());
        
        var A = Matrix<double>.Build.Dense(n, 2, 1.0);
        A.SetColumn(0, x);
        
        var p = A.QR().Solve(y);
        var residuals = (A * p - y).L2Norm();
        double error = Math.Pow(residuals, 2);
        
        return Tuple.Create(p.ToArray(), error);
    }

    public static double SumSquaredError(List<double> sequence, Segment segment)
    {
        var (p, error) = LeastSquaresLineFit(sequence, Tuple.Create((int)segment.X0, (int)segment.X1));
        return error;
    }

    public static Segment Regression(List<double> sequence, Tuple<int, int> seqRange)
    {
        var (p, error) = LeastSquaresLineFit(sequence, seqRange);
        double y0 = p[0] * seqRange.Item1 + p[1];
        double y1 = p[0] * seqRange.Item2 + p[1];
        return new Segment(seqRange.Item1, y0, seqRange.Item2, y1);
    }

    public static Segment Interpolate(List<double> sequence, Tuple<int, int> seqRange)
    {
        return new Segment(seqRange.Item1, sequence[seqRange.Item1], seqRange.Item2, sequence[seqRange.Item2]);
    }

    public static List<Segment> SlidingWindowSegment(List<double> sequence, 
        Func<List<double>, Tuple<int, int>, Segment> createSegment,
        Func<List<double>, Segment, double> computeError,
        double maxError, Tuple<int, int> seqRange = null)
    {
        if (seqRange == null)
            seqRange = Tuple.Create(0, sequence.Count - 1);

        int start = seqRange.Item1;
        int end = start;
        Segment resultSegment = createSegment(sequence, Tuple.Create(seqRange.Item1, seqRange.Item2));
        
        while (end < seqRange.Item2)
        {
            end++;
            Segment testSegment = createSegment(sequence, Tuple.Create(start, end));
            double error = computeError(sequence, testSegment);
            if (error <= maxError)
            {
                resultSegment = testSegment;
            }
            else
            {
                break;
            }
        }

        if (end == seqRange.Item2)
        {
            return new List<Segment> { resultSegment };
        }
        else
        {
            var rest = SlidingWindowSegment(sequence, createSegment, computeError, maxError, 
                Tuple.Create(end - 1, seqRange.Item2));
            return new List<Segment> { resultSegment }.Concat(rest).ToList();
        }
    }

    public static List<Segment> BottomUpSegment(
        List<double> sequence,
        Func<List<double>, Tuple<int, int>, Segment> createSegment,
        Func<List<double>, Segment, double> computeError,
        double maxError)
    {
        // Create initial segments (each point to its neighbor)
        var segments = Enumerable.Range(0, sequence.Count - 1)
            .Select(i => createSegment(sequence, Tuple.Create(i, i + 1)))
            .ToList();

        if (segments.Count < 2)
            return segments;

        // Precompute all possible merge segments and their costs
        var mergeSegments = new List<Segment>(segments.Count - 1);
        var mergeCosts = new List<double>(segments.Count - 1);

        for (int i = 0; i < segments.Count - 1; i++)
        {
            var merged = createSegment(sequence, Tuple.Create((int)segments[i].X0, (int)segments[i + 1].X1));
            mergeSegments.Add(merged);
            mergeCosts.Add(computeError(sequence, merged));
        }

        while (mergeCosts.Count > 0)
        {
            double minCost = mergeCosts.Min();
            if (minCost >= maxError)
                break;

            int idx = mergeCosts.IndexOf(minCost);

            // Merge the segments
            segments[idx] = mergeSegments[idx];
            segments.RemoveAt(idx + 1);

            // Update merge information
            mergeSegments.RemoveAt(idx);
            mergeCosts.RemoveAt(idx);

            // Update left neighbor if exists
            if (idx > 0)
            {
                mergeSegments[idx - 1] = createSegment(sequence, Tuple.Create((int)segments[idx - 1].X0, (int)segments[idx].X1));
                mergeCosts[idx - 1] = computeError(sequence, mergeSegments[idx - 1]);
            }

            // Update right neighbor if exists
            if (idx < mergeSegments.Count)
            {
                mergeSegments[idx] = createSegment(sequence, Tuple.Create((int)segments[idx].X0, (int)segments[idx + 1].X1));
                mergeCosts[idx] = computeError(sequence, mergeSegments[idx]);
            }
        }

        return segments;
    }

    public static List<Segment> TopDownSegment(List<double> sequence,
        Func<List<double>, Tuple<int, int>, Segment> createSegment,
        Func<List<double>, Segment, double> computeError,
        double maxError, Tuple<int, int> seqRange = null)
    {
        if (seqRange == null)
            seqRange = Tuple.Create(0, sequence.Count - 1);

        double bestLeftError = double.PositiveInfinity;
        double bestRightError = double.PositiveInfinity;
        Segment bestLeftSegment = null;
        Segment bestRightSegment = null;
        int bestIdx = 0;

        for (int idx = seqRange.Item1 + 1; idx < seqRange.Item2; idx++)
        {
            var segmentLeft = createSegment(sequence, Tuple.Create(seqRange.Item1, idx));
            double errorLeft = computeError(sequence, segmentLeft);
            var segmentRight = createSegment(sequence, Tuple.Create(idx, seqRange.Item2));
            double errorRight = computeError(sequence, segmentRight);

            if (errorLeft + errorRight < bestLeftError + bestRightError)
            {
                bestLeftError = errorLeft;
                bestRightError = errorRight;
                bestLeftSegment = segmentLeft;
                bestRightSegment = segmentRight;
                bestIdx = idx;
            }
        }

        List<Segment> leftSegs, rightSegs;

        if (bestLeftError <= maxError)
        {
            leftSegs = new List<Segment> { bestLeftSegment };
        }
        else
        {
            leftSegs = TopDownSegment(sequence, createSegment, computeError, maxError, 
                Tuple.Create(seqRange.Item1, bestIdx));
        }

        if (bestRightError <= maxError)
        {
            rightSegs = new List<Segment> { bestRightSegment };
        }
        else
        {
            rightSegs = TopDownSegment(sequence, createSegment, computeError, maxError, 
                Tuple.Create(bestIdx, seqRange.Item2));
        }

        return leftSegs.Concat(rightSegs).ToList();
    }
}
