// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.alg.ValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/* package */ enum GamblerHelper {
  ;
  static DiscreteQsa getOptimalQsa(GamblerModel gamblerModel) {
    return ActionValueIterations.solve(gamblerModel, RealScalar.of(.0001));
  }

  public static DiscreteVs getOptimalVs(GamblerModel gamblerModel) {
    return ValueIterations.solve(gamblerModel, RealScalar.of(1e-10));
  }

  public static Policy getOptimalPolicy(GamblerModel gamblerModel) {
    // TODO test for equality of policies from qsa and vs
    ValueIteration vi = new ValueIteration(gamblerModel, gamblerModel);
    vi.untilBelow(RealScalar.of(1e-10));
    return PolicyType.GREEDY.bestEquiprobable(gamblerModel, vi.vs(), null);
  }

  public static void play(GamblerModel gamblerModel, DiscreteQsa qsa) {
    DiscreteUtils.print(qsa, Round._2);
    System.out.println("---");
    Policy policy = PolicyType.GREEDY.bestEquiprobable(gamblerModel, qsa, null);
    EpisodeInterface mce = EpisodeKickoff.single(gamblerModel, policy, //
        gamblerModel.startStates().get(gamblerModel.startStates().length() / 2));
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
