// code by jph
package ch.ethz.idsc.subare.core.alg;

import java.util.Random;

import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** update similar to {@link QLearning} except does not use episodes
 * 
 * see box on p.169
 * 
 * algorithm performs poorly when rewards are unevenly distributed */
public class Random1StepTabularQPlanning {
  final StandardModel standardModel;
  final SampleModel sampleModel;
  Random random = new Random();
  final Tensor states;
  final QsaInterface qsa;
  final Scalar gamma;
  Scalar alpha;

  public Random1StepTabularQPlanning( //
      SampleModel sampleModel, StandardModel standardModel, //
      QsaInterface qsa, Scalar alpha) {
    this.sampleModel = sampleModel;
    this.standardModel = standardModel;
    states = standardModel.states();
    this.qsa = qsa;
    this.gamma = standardModel.gamma();
    this.alpha = alpha;
  }

  public void step() {
    Tensor state0 = states.get(random.nextInt(states.length()));
    Tensor actions = standardModel.actions(state0);
    Tensor action0 = actions.get(random.nextInt(actions.length()));
    Tensor state1 = sampleModel.move(state0, action0);
    Scalar reward = sampleModel.reward(state0, action0, state1);
    Scalar max = standardModel.actions(state1).flatten(0) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of).get();
    Scalar value0 = qsa.value(state0, action0);
    // TODO alpha should decrease over time
    Scalar delta = reward.add(gamma.multiply(max)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }
}
