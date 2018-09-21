// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class Sarsa_GamblerTest extends TestCase {
  public void testSimple() throws Exception {
    for (SarsaType sarsaType : SarsaType.values()) {
      Gambler gambler = Gambler.createDefault();
      gambler = new Gambler(20, RationalScalar.of(4, 10));
      Sarsa_Gambler.train(gambler, sarsaType, 10, RealScalar.of(3), RealScalar.of(0.81));
    }
  }
}
