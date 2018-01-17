// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class StepAdapterTest extends TestCase {
  @SuppressWarnings("unused")
  public void testSimple() {
    try {
      new StepAdapter(Tensors.empty(), Tensors.empty(), null, Tensors.empty());
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
