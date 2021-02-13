// code by jph
package ch.ethz.idsc.subare.ch02.bandits;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.api.TensorUnaryOperator;
import ch.ethz.idsc.tensor.nrm.Normalize;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.NormalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.ScalarSummaryStatistics;
import ch.ethz.idsc.tensor.red.StandardDeviation;
import ch.ethz.idsc.tensor.red.Variance;
import ch.ethz.idsc.tensor.sca.Chop;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Clips;

/** implementation corresponds to Figure 2.1, p. 30 */
/* package */ class Bandits {
  private static final TensorUnaryOperator NORMALIZE = Normalize.with(StandardDeviation::ofVector);
  private static final Distribution STANDARD = NormalDistribution.standard();
  // ---
  private final Tensor prep;

  public Bandits(int n) {
    Tensor data = RandomVariate.of(STANDARD, n);
    Scalar mean = (Scalar) Mean.of(data);
    prep = NORMALIZE.apply(data.map(x -> x.subtract(mean)));
    Chop._10.requireClose(Mean.of(prep), RealScalar.ZERO);
    Chop._10.requireClose(Variance.ofVector(prep), RealScalar.ONE);
  }

  private Scalar min = RealScalar.ZERO;
  private Scalar max = RealScalar.ZERO;

  Tensor pullAll() {
    Tensor states = prep.add(RandomVariate.of(STANDARD, prep.length()));
    ScalarSummaryStatistics scalarSummaryStatistics = //
        states.stream().map(Scalar.class::cast).collect(ScalarSummaryStatistics.collector());
    min = min.add(scalarSummaryStatistics.getMin());
    max = max.add(scalarSummaryStatistics.getMax());
    return states;
  }

  public Clip clip() {
    return Clips.interval(min, max);
  }
}
