// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.Map;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Tally;
import junit.framework.TestCase;

public class BlackjackTest extends TestCase {
  public void testSimple() {
    Blackjack blackjack = new Blackjack();
    {
      Tensor next = blackjack.move(Tensors.vector(0, 18, 7), RealScalar.ONE);
      // System.out.println(next);
    }
    {
      Tensor next = blackjack.move(Tensors.vector(0, 21, 7), RealScalar.ZERO);
      // System.out.println(next);
    }
  }

  public void testEpisodeLength() {
    Blackjack blackjack = new Blackjack();
    PolicyInterface pi = new EquiprobablePolicy(blackjack);
    Tensor tally = Tensors.empty();
    for (int EPISODES = 0; EPISODES < 10000; ++EPISODES) {
      EpisodeInterface ei = EpisodeKickoff.create(blackjack, pi);
      int count = 0;
      while (ei.hasNext()) {
        ei.step();
        ++count;
      }
      tally.append(RealScalar.of(count));
    }
    Map<Tensor, Long> map = Tally.of(tally);
    // {1=6574, 2=2537, 3=759, 4=121, 5=8, 7=1}
    assertTrue(5 <= map.size());
    // System.out.println("" + Tally.of(tally));
  }
}
