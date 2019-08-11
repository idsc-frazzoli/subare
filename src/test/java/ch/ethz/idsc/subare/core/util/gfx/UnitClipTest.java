// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class UnitClipTest extends TestCase {
  public void testSimple() {
    assertEquals(UnitClip.FUNCTION.apply(DoubleScalar.INDETERMINATE).toString(), DoubleScalar.INDETERMINATE.toString());
    assertEquals(UnitClip.FUNCTION.apply(DoubleScalar.POSITIVE_INFINITY), DoubleScalar.POSITIVE_INFINITY);
    assertEquals(UnitClip.FUNCTION.apply(RealScalar.of(3)), RealScalar.ONE);
  }
}
