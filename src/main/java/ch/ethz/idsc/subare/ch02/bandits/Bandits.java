// code by jph
package ch.ethz.idsc.subare.ch02.bandits;

import java.util.Random;

import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Sort;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.Variance;
import ch.ethz.idsc.tensor.sca.Chop;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** implementation corresponds to Figure 2.1, p. 30 */
class Bandits {
  public static final Random random = new Random();

  private static Tensor createGaussian(int n) {
    return Tensors.vector(i -> DoubleScalar.of(random.nextGaussian()), n);
  }

  // ---
  private final Tensor prep;
  private Tensor states;

  Bandits(int n) {
    Tensor data = createGaussian(n);
    Scalar mean = (Scalar) Mean.of(data);
    Tensor temp = data.map(x -> x.subtract(mean)).unmodifiable();
    prep = temp.multiply(Sqrt.function.apply((Scalar) Variance.ofVector(temp)).invert());
    GlobalAssert.of( //
        Chop.of(Mean.of(prep)).equals(ZeroScalar.get()));
    GlobalAssert.of( //
        Chop.of(Variance.ofVector(prep).subtract(RealScalar.of(1))) //
            .equals(ZeroScalar.get()));
  }

  Scalar min = ZeroScalar.get();
  Scalar max = ZeroScalar.get();

  void pullAll() {
    states = prep.add(createGaussian(prep.length()));
    Tensor sorted = Sort.of(states);
    min = min.add(sorted.Get(0));
    max = max.add(sorted.Get(states.length() - 1));
  }

  Scalar getLever(int k) {
    return states.Get(k);
  }
}
