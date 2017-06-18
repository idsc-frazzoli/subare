// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.PriorityQueue;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import junit.framework.TestCase;

public class PriorityQueueTest extends TestCase {
  public void testSimple() {
    PriorityQueue<Scalar> queue = new PriorityQueue<>();
    queue.add(RealScalar.of(3));
    queue.add(RealScalar.of(1));
    queue.add(RealScalar.of(10));
    queue.add(RealScalar.of(-9));
    assertEquals(queue.peek(), RealScalar.of(-9));
  }
}
