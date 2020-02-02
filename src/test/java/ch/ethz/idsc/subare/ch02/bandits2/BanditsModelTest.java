// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.Variance;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class BanditsModelTest extends TestCase {
  public void testMean() {
    int num = 10;
    BanditsModel banditsModel = new BanditsModel(num);
    Tensor means = Tensors.vector(k -> banditsModel.expectedReward(BanditsModel.START, RealScalar.of(k)), num);
    assertTrue(Chop._10.allZero(Mean.of(means)));
    Tensor starts = banditsModel.startStates();
    assertEquals(starts.length(), 1);
  }

  public void testExact() {
    int num = 20;
    BanditsModel banditsModel = new BanditsModel(num);
    DiscreteQsa ref = BanditsHelper.getOptimalQsa(banditsModel);
    Tensor expected = Tensors.vector(i -> ref.value(BanditsModel.START, RealScalar.of(i)), num);
    Scalar mean = Mean.of(expected).Get();
    assertTrue(Chop._10.allZero(mean));
    Scalar var = Variance.ofVector(expected).Get();
    assertTrue(Chop._10.close(var, RealScalar.ONE));
  }
}
