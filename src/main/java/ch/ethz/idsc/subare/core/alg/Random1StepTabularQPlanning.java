// code by jph
package ch.ethz.idsc.subare.core.alg;

import java.util.Random;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** update similar to {@link QLearning} except does not use episodes
 * 
 * see box on p.169
 * 
 * algorithm performs poorly when rewards are unevenly distributed */
public class Random1StepTabularQPlanning {
  private final DiscreteModel discreteModel;
  private final SampleModel sampleModel;
  private final Random random = new Random();
  private final DiscreteQsa qsa;
  private final Scalar gamma;
  private Scalar alpha = null;

  public Random1StepTabularQPlanning( //
      DiscreteModel discreteModel, SampleModel sampleModel, QsaInterface qsa) {
    this.discreteModel = discreteModel;
    this.sampleModel = sampleModel;
    this.qsa = (DiscreteQsa) qsa;
    this.gamma = discreteModel.gamma();
  }

  public void setUpdateFactor(Scalar alpha) {
    this.alpha = alpha;
  }

  public void step() {
    Tensor keys = qsa.keys();
    Tensor key = keys.get(random.nextInt(keys.length()));
    Tensor state0 = key.get(0); // TODO bypass if state0 is terminal?
    Tensor action0 = key.get(1);
    Tensor state1 = sampleModel.move(state0, action0);
    Scalar reward = sampleModel.reward(state0, action0, state1);
    Scalar max = discreteModel.actions(state1).flatten(0) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of).get();
    Scalar value0 = qsa.value(state0, action0);
    // TODO alpha should decrease over time
    Scalar delta = reward.add(gamma.multiply(max)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }
}
