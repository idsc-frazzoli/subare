// code by jph
package ch.ethz.idsc.subare.util;

import java.util.HashSet;
import java.util.Set;

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
}
