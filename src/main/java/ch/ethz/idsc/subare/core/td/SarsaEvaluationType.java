// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Max;

// TODO untested
public enum SarsaEvaluationType {
  ORIGINAL() {
    @Override
    Scalar evaluate(DiscreteModel discreteModel, Scalar epsilon, LearningRate learningRate, Tensor state, QsaInterface qsa) {
      Tensor actions = Tensor.of( //
          discreteModel.actions(state).stream() //
              .filter(action -> learningRate.encountered(state, action)));
      return Tensors.isEmpty(actions) ? RealScalar.ZERO : crossEvaluate(discreteModel, epsilon, state, actions, qsa, qsa);
    }

    @Override
    Scalar crossEvaluate(DiscreteModel discreteModel, Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2) {
      Policy policy = EGreedyPolicy.bestEquiprobable(discreteModel, qsa1, epsilon, state);
      Tensor action = new PolicyWrap(policy).next(state, actions);
      return qsa2.value(state, action);
    }
  }, //
  EXPECTED() {
    @Override
    Scalar evaluate(DiscreteModel discreteModel, Scalar epsilon, LearningRate learningRate, Tensor state, QsaInterface qsa) {
      Tensor actions = Tensor.of( //
          discreteModel.actions(state).stream() //
              .filter(action -> learningRate.encountered(state, action)));
      return Tensors.isEmpty(actions) ? RealScalar.ZERO : crossEvaluate(discreteModel, epsilon, state, actions, qsa, qsa);
    }

    @Override
    Scalar crossEvaluate(DiscreteModel discreteModel, Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2) {
      Policy policy = EGreedyPolicy.bestEquiprobable(discreteModel, qsa1, epsilon, state);
      return actions.stream() //
          .map(action -> policy.probability(state, action).multiply(qsa2.value(state, action))) //
          .reduce(Scalar::add).get();
    }
  }, //
  QLEARNING() {
    @Override
    Scalar evaluate(DiscreteModel discreteModel, Scalar epsilon, LearningRate learningRate, Tensor state, QsaInterface qsa) {
      return discreteModel.actions(state).stream() //
          .filter(action -> learningRate.encountered(state, action)) //
          .map(action -> qsa.value(state, action)) //
          .reduce(Max::of) //
          .orElse(RealScalar.ZERO);
    }

    @Override
    Scalar crossEvaluate(DiscreteModel discreteModel, Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2) {
      Scalar value = RealScalar.ZERO;
      Tensor eval = Tensor.of(actions.stream().map(action -> qsa1.value(state, action)));
      FairArgMax fairArgMax = FairArgMax.of(eval);
      Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount()); // uniform distribution among best actions
      for (int index : fairArgMax.options()) {
        Tensor action = actions.get(index);
        value = value.add(qsa2.value(state, action).multiply(weight)); // use Qsa2 to evaluate state-action pair
      }
      return value;
    }
  }, //
  ;
  abstract Scalar evaluate(DiscreteModel discreteModel, Scalar epsilon, LearningRate learningRate, Tensor state, QsaInterface qsa);

  abstract Scalar crossEvaluate(DiscreteModel discreteModel, Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2);
}
