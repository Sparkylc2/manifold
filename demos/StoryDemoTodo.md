


- add colour bar to the cylinder and fea flutter


## Rocket landing
- fix colours for rocket landing (both bg and thrust)
- make the PID controller look more compact and less thrown together
- move it down a bit so we can have text at the top
- fix the pid tuning so it lands a little faster without that oscillation
- have it use a far simpler drawing for both body and thruster
- fix clipping of thruster into ground
- make the bend faster (but not so fast it immediately goes to max extent)
- increase colour bar size for temp and stress so that they are actually visible



# 1.
## jansen
- done
## cart kickup
- done
## engine
- replace with the crane one or the metronome one

# 2
## fea flutter
- looks good, but explodes at higher mesh sizes on the FEA
- fix colouring sensitivity once recording settings have been set
## foil flutter
- fix the damping, the vortex issue doesn't seem to be very solveable, so instead damping increased
## nozzle
- definitely in the video have it sped up a bit, increase res, see if we can improve it a little


# 3
## narration
> Once the state is that small, you can learn how it evolves, and decode back
> out. Watch the echo state network and an LSTM predict the next 2.5 seconds of
> the flow. Neither stays correct forever, and neither could. In a chaotic
> system, any error grows exponentially (the largest Lyapunov exponent is
> positive).

The 2.5 s is exact, not a round number: `ForecastKarmanDemo::HORIZON` is 50
rollout steps at `SNAP_DT` = 0.05 s. Changing either breaks the line.

## pod 
- good, but add the opacity thing we have for the fea and the foil
## esn + lstm
- we will have both fully trained and fitted properly, and show a live reconstruction going forward however many seconds, as well as a reconstruction error plot
- will run next to eachother at the same time, just do what the current esn demo does


# 4
wave pde plot, maybe a vector field plot with attractors or some other weird math object but rendered beautifully, and then maybe something else
 

