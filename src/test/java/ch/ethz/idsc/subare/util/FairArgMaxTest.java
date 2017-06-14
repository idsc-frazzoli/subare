// code by jph
package ch.ethz.idsc.subare.util;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class FairArgMaxTest extends TestCase {
  public void testIsFair() {
    Tensor d = Tensors.vectorDouble(3, .3, 3, .6, 3);
    Set<Integer> set = new HashSet<>();
    FairArgMax fairArgMax = FairArgMax.of(d);
    for (int index = 0; index < 100; ++index)
      set.add(fairArgMax.nextRandomIndex());
    assertEquals(set.size(), 3);
  }

  public void testEmpty() {
    try {
      FairArgMax.of(Tensors.empty());
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testInfty() {
    Tensor d = Tensors.of( //
        RealScalar.POSITIVE_INFINITY, RealScalar.ONE, //
        RealScalar.POSITIVE_INFINITY, RealScalar.POSITIVE_INFINITY);
    FairArgMax fairArgMax = FairArgMax.of(d);
    assertEquals(fairArgMax.optionsCount(), 3);
    List<Integer> list = fairArgMax.options();
    assertEquals(list, Arrays.asList(0, 2, 3));
  }
}
