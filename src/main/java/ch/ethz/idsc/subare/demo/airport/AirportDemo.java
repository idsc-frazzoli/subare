// code by fluric
package ch.ethz.idsc.subare.demo.airport;

import java.util.Arrays;

import ch.ethz.idsc.subare.analysis.DiscreteModelErrorAnalysis;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.td.OriginalTrueOnlineSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.StateActionCounter;
import ch.ethz.idsc.subare.util.PlotUtils;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** uses TrueOnlineSarsa */
enum AirportDemo {
  ;
  public static void main(String[] args) throws Exception {
    Tensor XYmc = Tensors.empty();
    Tensor XYsarsa = Tensors.empty();
    Tensor XYtoSarsa = Tensors.empty();
    Airport airport = new Airport();
    DiscreteQsa optimalQsa = ActionValueIterations.solve(airport, DecimalScalar.of(.0001));
    // DiscreteUtils.print(optimalQsa);
    Policy policyQsa = GreedyPolicy.bestEquiprobable(airport, optimalQsa);
    // Policies.print(policyQsa, airport.states());
    final int batches = 10;
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(airport);
    {
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        Policy policyMC = EGreedyPolicy.bestEquiprobable(airport, mces.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(airport, policyMC, mces);
        XYmc.append(Tensors.vector(RealScalar.of(index).number(), DiscreteModelErrorAnalysis.LINEAR_POLICY.getError(airport, optimalQsa, mces.qsa()).number()));
      }
      System.out.println("time for MonteCarlo: " + stopwatch.display_seconds() + "s");
      // Policies.print(GreedyPolicy.bestEquiprobable(airport, mces.qsa()), airport.states());
    }
    StateActionCounter sac = new StateActionCounter(airport);
    DiscreteQsa qsaSarsa = DiscreteQsa.build(airport); // q-function for training, initialized to 0
    SarsaType sarsaType = SarsaType.ORIGINAL;
    final Sarsa sarsa = sarsaType.supply(airport, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
    {
      sarsa.setExplore(RealScalar.of(.1));
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        Policy policy = EGreedyPolicy.bestEquiprobable(airport, sarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(airport, policy, 1, sarsa, sac);
        XYsarsa.append(
            Tensors.vector(RealScalar.of(index).number(), DiscreteModelErrorAnalysis.LINEAR_POLICY.getError(airport, optimalQsa, sarsa.qsa()).number()));
      }
      System.out.println("time for Sarsa: " + stopwatch.display_seconds() + "s");
    }
    // Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
    LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.2));
    FeatureMapper mapper = ExactFeatureMapper.of(airport);
    TrueOnlineSarsa toSarsa = OriginalTrueOnlineSarsa.of(airport, RealScalar.of(0.7), learningRate, mapper);
    toSarsa.setExplore(RealScalar.of(0.1));
    {
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        Policy policy = EGreedyPolicy.bestEquiprobable(airport, toSarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(airport, policy, toSarsa);
        DiscreteQsa toQsa = toSarsa.qsa();
        XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), DiscreteModelErrorAnalysis.LINEAR_POLICY.getError(airport, optimalQsa, toQsa).number()));
      }
      System.out.println("time for TrueOnlineSarsa: " + stopwatch.display_seconds() + "s");
    }
    DiscreteQsa toQsa = toSarsa.qsa();
    // System.out.println(toSarsa.getW());
    // toSarsa.printValues();
    // toSarsa.printPolicy();
    PlotUtils.createPlot(Arrays.asList(XYmc, XYsarsa, XYtoSarsa), Arrays.asList("MonteCarlo", "Sarsa", "TrueOnlineSarsa"), "Convergence_Airport");
  }
}
