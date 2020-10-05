// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.ethz.idsc.subare.util.AssertFail;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class StepAdapterTest extends TestCase {
  public void testSimple() {
    AssertFail.of(() -> new StepAdapter(Tensors.empty(), Tensors.empty(), null, Tensors.empty()));
  }
}
