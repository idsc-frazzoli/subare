// code by jph
package ch.ethz.idsc.subare.util;

import java.util.HashSet;
import java.util.Set;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class FairArgTest extends TestCase {
  public void testIsFair() {
    Tensor d = Tensors.vectorDouble(3, .3, 3, .6, 3);
    Set<Integer> set = new HashSet<>();
    for (int index = 0; index < 100; ++index)
      set.add(FairArg.max(d));
    assertEquals(set.size(), 3);
  }
}
