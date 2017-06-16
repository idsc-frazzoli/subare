// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class BanditsTest extends TestCase {
  public void testMean() {
    int num = 10;
    Bandits bandits = new Bandits(num);
    Tensor means = Tensors.vector(k -> bandits.expectedReward(Bandits.START, RealScalar.of(k)), num);
    assertTrue(Chop.isZeros(Mean.of(means)));
  }
}
