// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class Sarsa_GamblerTest extends TestCase {
  public void testSimple() throws Exception {
    for (SarsaType sarsaType : SarsaType.values()) {
      GamblerModel gamblerModel = new GamblerModel(20, RationalScalar.of(4, 10));
      Sarsa_Gambler sarsa_Gambler = new Sarsa_Gambler(gamblerModel);
      LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
      sarsa_Gambler.train(sarsaType, 10, learningRate);
    }
  }
}
