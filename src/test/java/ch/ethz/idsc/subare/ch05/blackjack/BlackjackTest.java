// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import junit.framework.TestCase;

public class BlackjackTest extends TestCase {
  public void testSimple() {
    Blackjack blackjack = new Blackjack();
    {
      Tensor next = blackjack.move(Tensors.vector(0, 18, 7), RealScalar.ONE);
      System.out.println(next);
    }
    {
      Tensor next = blackjack.move(Tensors.vector(0, 21, 7), ZeroScalar.get());
      System.out.println(next);
    }
  }
}
