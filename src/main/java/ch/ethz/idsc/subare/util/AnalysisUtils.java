package ch.ethz.idsc.subare.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.StateActionCounter;
import ch.ethz.idsc.subare.demo.airport.Airport;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Abs;

public class AnalysisUtils {
  public static Scalar getLinearQsaError(DiscreteQsa refQsa, DiscreteQsa currentQsa) {
    GlobalAssert.that(refQsa.size() == currentQsa.size());
    Scalar error = RealScalar.ZERO;
    Scalar delta;
    for (int index = 0; index < refQsa.size(); ++index) {
      delta = Abs.of(refQsa.values().get(index).subtract(currentQsa.values().get(index))).Get();
      error = error.add(delta);
    }
    return error;
  }

  public static Scalar getSquareQsaError(DiscreteQsa refQsa, DiscreteQsa currentQsa) {
    GlobalAssert.that(refQsa.size() == currentQsa.size());
    Scalar error = RealScalar.ZERO;
    Scalar delta;
    for (int index = 0; index < refQsa.size(); ++index) {
      delta = Abs.of(refQsa.values().get(index).subtract(currentQsa.values().get(index))).Get();
      error = error.add(Times.of(delta, delta));
    }
    return error;
  }

  public static void analyse(MonteCarloInterface mcInterface, int batches) throws Exception {
    DiscreteQsa optimalQsa = getOptimalQsa(mcInterface);
    List<Tensor> data = new ArrayList<>();
    List<String> names = new ArrayList<>();
    List<List<Object>> data_name = new ArrayList<>();
    // ---
    data_name.add(analyseMonteCarlo(mcInterface, batches, optimalQsa));
    data_name.add(analyseSarsa(mcInterface, batches, optimalQsa));
    data_name.add(analyseTrueOnlineSarsa(mcInterface, batches, optimalQsa));
    // ---
    for (List<Object> set : data_name) {
      data.add((Tensor) set.get(0));
      names.add((String) set.get(1));
    }
    PlotUtils.createPlot(data, names, "Convergence_" + mcInterface.getClass().getSimpleName().toString());
  }

  public static DiscreteQsa getOptimalQsa(MonteCarloInterface mcInterface) {
    long start = System.currentTimeMillis();
    DiscreteQsa optimalQsa = ActionValueIterations.solve((StandardModel) mcInterface, DecimalScalar.of(.0001));
    System.out.println("time for AVI: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
    // DiscreteUtils.print(optimalQsa);
    // Policy policyQsa = GreedyPolicy.bestEquiprobable(mcInterface, optimalQsa);
    // Policies.print(policyQsa, airport.states());
    return optimalQsa;
  }

  public static List<Object> analyseSarsa(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
    Tensor XYsarsa = Tensors.empty();
    StateActionCounter sac = new StateActionCounter(mcInterface);
    DiscreteQsa qsaSarsa = DiscreteQsa.build(mcInterface); // q-function for training, initialized to 0
    final Sarsa sarsa = new OriginalSarsa(mcInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
    sarsa.setExplore(RealScalar.of(.1));
    long start = System.currentTimeMillis();
    for (int index = 0; index < batches; ++index) {
      // System.out.println("starting batch " + (index + 1) + " of " + batches);
      Policy policy = EGreedyPolicy.bestEquiprobable(mcInterface, sarsa.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(mcInterface, policy, 1, sarsa, sac);
      XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number()));
    }
    System.out.println("Time for Sarsa: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
    System.out.println("Error of Sarsa: Linear: " + AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number().doubleValue() + " Quadratic: "
        + AnalysisUtils.getSquareQsaError(sarsa.qsa(), optimalQsa).number().doubleValue());
    // Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
    return Arrays.asList(XYsarsa, "Sarsa");
  }

  public static List<Object> analyseMonteCarlo(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
    Tensor XYmc = Tensors.empty();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(mcInterface);
    long start = System.currentTimeMillis();
    for (int index = 0; index < batches; ++index) {
      // System.out.println("starting batch " + (index + 1) + " of " + batches);
      Policy policyMC = EGreedyPolicy.bestEquiprobable(mcInterface, mces.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(mcInterface, policyMC, mces);
      XYmc.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(mces.qsa(), optimalQsa).number()));
    }
    System.out.println("Time for MonteCarlo: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
    System.out.println("Error of MonteCarlo: Linear:" + AnalysisUtils.getLinearQsaError(mces.qsa(), optimalQsa).number().doubleValue() + " Quadratic: "
        + AnalysisUtils.getSquareQsaError(mces.qsa(), optimalQsa).number().doubleValue());
    // Policies.print(GreedyPolicy.bestEquiprobable(airport, mces.qsa()), airport.states());
    return Arrays.asList(XYmc, "MonteCarlo");
  }

  public static List<Object> analyseTrueOnlineSarsa(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
    Tensor XYtoSarsa = Tensors.empty();
    FeatureMapper mapper = new ExactFeatureMapper(mcInterface);
    TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(mcInterface, RealScalar.of(0.7), RealScalar.of(0.2), RealScalar.of(1), mapper);
    long start = System.currentTimeMillis();
    for (int index = 0; index < batches; ++index) {
      // System.out.println("starting batch " + (index + 1) + " of " + batches);
      toSarsa.executeEpisode(RealScalar.of(0.1));
      DiscreteQsa toQsa = toSarsa.getQsa();
      XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(toQsa, optimalQsa).number()));
    }
    System.out.println("Time for TrueOnlineSarsa: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
    DiscreteQsa toQsa = toSarsa.getQsa();
    // System.out.println(toSarsa.getW());
    // toSarsa.printValues();
    // toSarsa.printPolicy();
    System.out.println("Error of TrueOnlineSarsa: Linear: " + AnalysisUtils.getLinearQsaError(toQsa, optimalQsa).number().doubleValue() + " Quadratic: "
        + AnalysisUtils.getSquareQsaError(toQsa, optimalQsa).number().doubleValue());
    return Arrays.asList(XYtoSarsa, "TrueOnlineSarsa");
  }

  public static void main(String[] args) throws Exception {
    // MonteCarloInterface mcInterface = Gambler.createDefault();
    // MonteCarloInterface mcInterface = DynamazeHelper.create5(3);
    MonteCarloInterface mcInterface = new Airport();
    // ---
    analyse(mcInterface, 100000);
    // analyseTrueOnlineSarsa(mcInterface, 10, getOptimalQsa(mcInterface));
  }
}
