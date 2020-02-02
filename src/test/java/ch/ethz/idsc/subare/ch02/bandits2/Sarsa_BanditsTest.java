// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Chop;
import ch.ethz.idsc.tensor.sca.Sign;
import junit.framework.TestCase;

public class Sarsa_BanditsTest extends TestCase {
  public void testSimple() {
    BanditsModel banditsModel = new BanditsModel(10);
    BanditsTrain sarsa_Bandits = new BanditsTrain(banditsModel);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(16), RealScalar.of(1.15));
    DiscreteQsa qsa = sarsa_Bandits.train(SarsaType.ORIGINAL, 100, learningRate);
    DiscreteVs rvs = DiscreteUtils.createVs(banditsModel, sarsa_Bandits.ref);
    DiscreteVs cvs = DiscreteUtils.createVs(banditsModel, qsa);
    Sign.requirePositive(rvs.value(RealScalar.ZERO));
    Chop.NONE.requireAllZero(cvs.value(RealScalar.ONE));
  }
}
