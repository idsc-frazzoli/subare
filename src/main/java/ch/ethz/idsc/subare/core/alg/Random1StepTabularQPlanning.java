// code by jph
package ch.ethz.idsc.subare.core.alg;

import java.util.Random;

import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** see box on p.169 */
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
      QsaInterface qsa, Scalar gamma, Scalar alpha) {
    this.sampleModel = sampleModel;
    this.standardModel = standardModel;
    states = standardModel.states();
    this.qsa = qsa;
    this.gamma = gamma;
    this.alpha = alpha;
  }

  // TODO this has not been tested
  void step() {
    Tensor state = states.get(random.nextInt(states.length()));
    Tensor actions = standardModel.actions(state);
    Tensor action = actions.get(random.nextInt(actions.length()));
    Tensor stateP = sampleModel.move(state, action);
    Scalar reward = sampleModel.reward(state, action, stateP); // deterministic
    Scalar max = standardModel.actions(stateP).flatten(0) //
        .map(a -> qsa.value(stateP, a)) //
        .reduce(Max::of).get();
    Scalar value0 = qsa.value(state, action);
    // TODO alpha should decrease over time
    Scalar delta = alpha.multiply(reward.add(gamma.multiply(max)).subtract(value0));
    qsa.assign(state, action, value0.add(delta));
  }
}
