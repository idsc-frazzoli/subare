// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.opt.Interpolation;
import junit.framework.TestCase;

public class ColorschemeTest extends TestCase {
  public void testSimple() {
    Interpolation interpolation = Colorscheme.classic();
    assertTrue(interpolation.get(Tensors.vector(200, 1)).isScalar());
  }

  public void testTensor027() throws Exception {
    // jar:file:/home/datahaki/.m2/repository/ch/ethz/idsc/tensor/0.2.7/tensor-0.2.7.jar!/colorscheme/classic.csv
    // URL url = ResourceData.url("/colorscheme/classic.csv");
    // System.out.println(url);
    // // jar:file:/home/datahaki/.m2/repository/ch/ethz/idsc/tensor/0.2.7/tensor-0.2.7.jar!/colorscheme/classic.csv
    // URI uri = ResourceData.uri("/colorscheme/classic.csv");
    // System.out.println(uri);
    // System.out.println(uri.getPath());
    // Tensor tensor = ResourceData.of("/colorscheme/classic.csv");
    // System.out.println(tensor);
  }
}
