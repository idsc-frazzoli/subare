// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.opt.Interpolation;
import junit.framework.TestCase;

public class ColorschemeTest extends TestCase {
  public void testSimple() {
    Interpolation interpolation = Colorscheme.CLASSIC;
    assertTrue(interpolation.get(Tensors.vector(200, 1)).isScalar());
  }
}
