// code by jph
package ch.ethz.idsc.subare.util;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Serialization;
import junit.framework.TestCase;

public class FairArgTest extends TestCase {
  public void testMaxIsFair() throws ClassNotFoundException, IOException {
    Tensor d = Tensors.vectorDouble(3, .3, 3, .6, 3);
    Set<Integer> set = new HashSet<>();
    FairArg fairArg = Serialization.copy(FairArg.max(d));
    for (int index = 0; index < 100; ++index)
      set.add(fairArg.nextRandomIndex());
    assertEquals(set.size(), 3);
  }

  public void testMinIsFair() {
    Tensor d = Tensors.vectorDouble(3, .3, 3, .6, 3, .3);
    Set<Integer> set = new HashSet<>();
    FairArg fairArg = FairArg.min(d);
    for (int index = 0; index < 100; ++index)
      set.add(fairArg.nextRandomIndex());
    assertEquals(set.size(), 2);
  }

  public void testInfty() {
    Tensor d = Tensors.of( //
        DoubleScalar.POSITIVE_INFINITY, RealScalar.ONE, //
        DoubleScalar.POSITIVE_INFINITY, DoubleScalar.POSITIVE_INFINITY);
    FairArg fairArg = FairArg.max(d);
    assertEquals(fairArg.optionsCount(), 3);
    List<Integer> list = fairArg.options();
    assertEquals(list, Arrays.asList(0, 2, 3));
  }

  public void testEmptyFail() {
    try {
      FairArg.max(Tensors.empty());
      fail();
    } catch (Exception exception) {
      // ---
    }
  }

  public void testScalarFail() {
    try {
      FairArg.max(RealScalar.ONE);
      fail();
    } catch (Exception exception) {
      // ---
    }
  }
}
